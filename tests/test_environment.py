import sys
import torch
import numpy as np

def test_python_version():
    """Verify that Python is running and satisfies basic requirements (>= 3.8)."""
    assert sys.version_info >= (3, 8), f"Python version is too old: {sys.version_info}"
    print(f"\n[Test] Python version: {sys.version}")

def test_pytorch_import():
    """Verify that PyTorch can be successfully imported and versions are checkable."""
    assert torch.__version__ is not None
    print(f"[Test] PyTorch version: {torch.__version__}")

def test_numpy_import():
    """Verify that NumPy can be successfully imported."""
    assert np.__version__ is not None
    print(f"[Test] NumPy version: {np.__version__}")

def test_cpu_tensor():
    """Verify that a basic CPU tensor can be created and operations run."""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert tensor.device.type == 'cpu'
    assert torch.equal(tensor + 1.0, torch.tensor([2.0, 3.0, 4.0]))
    print(f"[Test] Created CPU tensor: {tensor} on device: {tensor.device}")

def test_cuda_detection():
    """Detect CUDA availability without forcing its usage and output details."""
    cuda_available = torch.cuda.is_available()
    print(f"\n[Test] CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"[Test] CUDA Device Count: {device_count}")
        print(f"[Test] CUDA Device Name: {device_name}")
        assert device_count > 0
    else:
        print("[Test] CUDA is not available. Using CPU-only operations.")
        device = torch.device("cuda" if cuda_available else "cpu")
        assert device.type == 'cpu'
