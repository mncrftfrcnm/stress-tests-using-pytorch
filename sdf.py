import torch

num_cuda_devices = torch.cuda.device_count()
print(f"Number of CUDA devices available: {num_cuda_devices}")
