# quick_cuda_test.py
import torch
print(f"CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}, Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
