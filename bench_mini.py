import torch
import time
import json
import os
import evaluate
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# THE ROSTER
# "type" determines the prompt format:
# - "meta": Uses System + User prompts (Llama, Qwen, Mistral)
# - "google": Uses User prompt only (Gemma crashes with System prompts)
POTENTIAL_MODELS = [
    { "name": "Qwen-2.5-3B",  "path": "./models/qwen-2.5-3b", "type": "meta" },
    { "name": "Gemma-2-2B",   "path": "./models/gemma-2-2b",  "type": "google" },
    { "name": "Llama-3.2-3B", "path": "./models/llama-3.2-3b", "type": "meta" },
]

# Filter list to only models that are actually downloaded
MODELS_TO_TEST = [m for m in POTENTIAL_MODELS if os.path.exists(m["path"])]

if not MODELS_TO_TEST:
    print("No models found. Please check your ./models folder.")
    exit()

print(f"Found {len(MODELS_TO_TEST)} models to benchmark: {[m['name'] for m in MODELS_TO_TEST]}")

# Load Metrics
print("Loading Metrics...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")

final_report = {}

for model_info in MODELS_TO_TEST:
    model_name = model_info["name"]
    model_path = model_info["path"]
    
    print("\n" + "="*50)
    print(f"  STARTING: {model_name} (High Precision FP16)")
    print("="*50)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We use torch.float16 directly (No compression needed for these sizes)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,    # Pure 16-bit Precision
        device_map="cuda:0",          # Force GPU
        local_files_only=True,
    )

    results = []
    references = []
    predictions = []

    for item in tqdm(dataset, desc=f"Benchmarking {model_name}"):
        article = item['article']
        true_summary = item['highlights']

        # HANDLE PROMPT DIFFERENCES based on "type"
        if model_info["type"] == "google":
            # Gemma Style (No System Prompt allowed)
            messages = [
                {"role": "user", "content": f"Summarize the following article in 3 sentences:\n\n{article}"}
            ]
        else:
            # Standard Style (System Prompt supported)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following article in 3 sentences:\n\n{article}"}
            ]

        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to("cuda")
        
        # Create Mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to("cuda")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)

        start_time = time.time()
        thread.start()

        generated_text = ""
        first_token_received = False
        ttft = 0.0

        for new_text in streamer:
            if not first_token_received:
                ttft = time.time() - start_time
                first_token_received = True
            generated_text += new_text

        end_time = time.time()

        # Metrics
        latency = end_time - start_time
        output_tokens = len(tokenizer.encode(generated_text))
        speed_tps = output_tokens / latency if latency > 0 else 0
        tpot = (latency- ttft) / (output_tokens-1) if output_tokens > 1 else 0
        
        predictions.append(generated_text)
        references.append(true_summary)
        
        results.append({
            "latency": latency,
            "speed_tps": speed_tps,
            "ttft": ttft,
            "tpot": tpot
        })

    # Free Memory for Next Loop
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Compute Scores
    print(f"Computing BERTScore for {model_name}...")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_res = bleu.compute(predictions=predictions, references=references)
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en", device="cpu")
    avg_bert = np.mean(bert_res['f1'])
    avg_tps = np.mean([r['speed_tps'] for r in results])
    
    print(f"-> {model_name}: {avg_tps:.2f} t/s | BERT: {avg_bert:.4f}")
    
    final_report[model_name] = {
        "throughput": avg_tps,
        "bert_score": avg_bert,
        "runs": results,
        "rouge": rouge_score,
        "bleu": bleu_res
    }

# Save All
with open("benchmark_results_tiny_giants.json", "w") as f:
    json.dump(final_report, f, indent=2)

print("\n--- ALL TINY MODELS FINISHED ---")