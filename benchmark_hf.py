import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Define your local paths clearly
# Make sure your folder structure looks like:
# project_folder/
# └── models/
#     ├── llama3-8b/  (contains config.json, *.safetensors)
#     ├── mistral-7b/
#     └── gemma-2-9b/
local_model_path = "./models/llama3" 

# Check if path exists to avoid confusing errors
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"Could not find model at {local_model_path}")

print(f"Loading model from: {local_model_path}...")

# 2. Configure 4-bit Quantization (CRITICAL for 8GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Fix for Llama 3 / Mistral (they often lack a pad token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Load the Model
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config, # <--- This prevents the crash
    device_map="auto",              # Automatically uses your GPU
    local_files_only=True           # Forces it to look ONLY offline
)

print(f"Successfully loaded {local_model_path}!")

# 5. Define a prompt (Benchmarking style)
input_text = "The key difference between a CPU and a GPU is"

# Apply chat template if it's an instruct model (Optional but recommended)
# input_text = f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 6. Generate Response
print("Generating...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=50, 
    pad_token_id=tokenizer.pad_token_id
)

# 7. Decode and Print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Model Output ---")
print(response)