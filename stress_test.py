import torch
import time
import threading

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform matrix multiplication
def matrix_multiplication_stress(size, duration):
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    start_time = time.time()

    while time.time() - start_time < duration:
        torch.matmul(A, B)

# Function to perform stress test using multiple threads
def gpu_stress_test(duration=120, num_threads=4, matrix_size=2048):
    threads = []
    print(f"Starting GPU stress test for {duration} seconds with {num_threads} threads...")

    # Create and start threads
    for _ in range(num_threads):
        thread = threading.Thread(target=matrix_multiplication_stress, args=(matrix_size, duration))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(f"Completed GPU stress test for {duration} seconds with {num_threads} threads")

# Run the stress test
gpu_stress_test()