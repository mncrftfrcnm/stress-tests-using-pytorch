import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform stress test for 2 minutes
def gpu_stress_test(duration=120):
    # Create two random matrices
    size = 2048
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(10):
        torch.matmul(A, B)
    
    # Start the stress test
    print(f"Starting GPU stress test for {duration} seconds...")
    start_time = time.time()
    operations = 0

    while time.time() - start_time < duration:
        torch.matmul(A, B)
        operations += 1
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Completed {operations} matrix multiplications in {total_time:.2f} seconds")

# Run the stress test
gpu_stress_test()
