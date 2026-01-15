import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print("✅ SUCCESS: Your GPU is linked to Python!")
else:
    print("❌ ERROR: Python cannot see your GPU yet.")
    