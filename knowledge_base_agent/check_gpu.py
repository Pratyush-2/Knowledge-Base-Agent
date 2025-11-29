import torch

def check_gpu_health():
    """
    Checks the status of PyTorch, CUDA, and GPU availability.
    """
    print("--- PyTorch GPU Health Check ---")
    try:
        # 1. PyTorch Version
        print(f"PyTorch Version: {torch.__version__}")

        # 2. CUDA Availability
        is_available = torch.cuda.is_available()
        print(f"CUDA Available: {is_available}")

        if not is_available:
            print("\n[FAIL] PyTorch cannot detect a CUDA-enabled GPU.")
            print("This is the root cause of the error in the application.")
            print("Please ensure your NVIDIA drivers are correctly installed and that you installed the CUDA-enabled version of PyTorch.")
        else:
            # 3. Number of GPUs
            device_count = torch.cuda.device_count()
            print(f"Number of GPUs: {device_count}")

            # 4. CUDA Version PyTorch was built with
            print(f"PyTorch built with CUDA Version: {torch.version.cuda}")

            # 5. Current GPU Details
            for i in range(device_count):
                print(f"\n--- GPU {i} ---")
                print(f"Name: {torch.cuda.get_device_name(i)}")
                print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")

            print("\n[SUCCESS] PyTorch is correctly configured to use your GPU.")

    except Exception as e:
        print(f"\n[ERROR] An exception occurred during the check: {e}")

    print("\n--- End of Health Check ---")


if __name__ == "__main__":
    check_gpu_health()