from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

# The Tiny Contenders
models = [
    {"repo": "meta-llama/Llama-3.2-3B-Instruct", "dir": "./models/llama-3.2-3b"},
    {"repo": "google/gemma-2-2b-it",             "dir": "./models/gemma-2-2b"},
    {"repo": "Qwen/Qwen2.5-3B-Instruct",         "dir": "./models/qwen-2.5-3b"}
]

for m in models:
    print(f"\nDownloading {m['repo']}...")
    snapshot_download(
        repo_id=m["repo"],
        local_dir=m["dir"],
        local_dir_use_symlinks=False
    )

print("\nAll Tiny Giants downloaded!")