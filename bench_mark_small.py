import torch
import time
import json
import os
import evaluate
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
model_path = "./models/phi-3.5-mini-instruct"
output_file = "benchmark_results_phi3_mini_instruct_pytorch.json"

# Load Metrics
print("Loading Metrics (ROUGE, BERTScore, BLEU)...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

# 2. Load Model
print(f"Loading Model from {model_path}...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,              # <--- Changed to 8-bit (Higher Precision)
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False) #for microsoft models
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="cuda:0",
    local_files_only=True,
    trust_remote_code=False #for microsoft models
)

# 3. Load Dataset
print("Loading Test Data...")
# Running 5 samples
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")

print("Benchmarking...")
results=[]
references=[]
predictions=[]

for item in tqdm(dataset):
    article = item['article']
    true_summary = item['highlights']
    
    # Prompt
#phi 3.5 mini instruct
    messages = [
        { "role":"system", "content": "You are a helpful assistant that summarizes news articles." },
        { "role":"user", "content": f"Summarize the following article in 3 sentences: \n\n{article}" }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs=dict(
        input_ids = input_ids,
        max_new_tokens=150,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    start_time = time.time()
    thread.start()

    generate_text=""
    first_token_recieved = False
    ttft=0.0

    for new_test in streamer:
        if not first_token_recieved:
            ttft = time.time() - start_time
            first_token_recieved = True
        generate_text += new_test

    end_time = time.time()

    #metrics
    latency = end_time - start_time
    output_tokens = len(tokenizer.encode(generate_text))
    speed_tps = output_tokens / latency if latency > 0 else 0
    tpot = (latency-ttft) / (output_tokens-1) if output_tokens > 1 else 0

    predictions.append(generate_text)
    references.append(true_summary)

    results.append({
        "generation": generate_text,
        "latency": latency,
        "speed_tps": speed_tps,
        "ttft": ttft,
        "tpot": tpot

    })

# Overall Metrics
print(f"Computing final scores...")
rouge_res = rouge.compute(predictions=predictions, references=references)
bert_res = bertscore.compute(predictions=predictions, references=references, lang="en", device="cpu")
bleu_res = bleu.compute(predictions=predictions, references=references)
avg_bert = np.mean(bert_res['f1'])

# 5. Final Averages
avg_speed = np.mean([r['speed_tps'] for r in results])
avg_ttft = np.mean([r['ttft'] for r in results])
avg_tpot = np.mean([r['tpot'] for r in results])
avg_latency = np.mean([r['latency'] for r in results])
avg_bert = np.mean(bert_res['f1'])

print("\n" + "="*50)
print(f"  BENCHMARK REPORT: Llama 3 (8B) - RTX 4060")
print("="*50)
print(f"Avg Latency:    {avg_latency:.4f} s")
print(f"Avg TTFT:       {avg_ttft:.4f} s")
print(f"Avg TPOT:       {avg_tpot:.4f} s")
print(f"Throughput:     {avg_speed:.2f} tokens/sec")
print("-" * 50)
print(f"ROUGE-L:        {rouge_res['rougeL']:.4f}")
print(f"BERTScore F1:   {avg_bert:.4f}")
print(f"BLEU:           {bleu_res['bleu']:.4f}")
print("="*50)

# Save to file
output_data = {
    "summary": {
        "model": "Llama-3-8B-Instruct", #update according to model
        "latency": avg_latency,    # <--- Renamed to simple keys
        "ttft": avg_ttft,
        "tpot": avg_tpot,
        "throughput": avg_speed,
        "rouge": rouge_res,
        "bleu": bleu_res,
        "bert_score_f1": avg_bert
    },
    "detailed_runs": results
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Detailed results saved to {output_file}")