import pytest
import torch

def test_gemm_quantized_kernel():
    """Verifies correctness of custom CUDA quantized GEMM against PyTorch reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
        
    import llm_engine_cuda

    m, n, k = 4, 16, 32
    input_tensor = torch.randn((m, k), dtype=torch.float32, device="cuda")
    weight_tensor = torch.randint(0, 255, (k, n), dtype=torch.uint8, device="cuda")
    scales = torch.ones((n,), dtype=torch.float32, device="cuda")

    # Run custom CUDA kernel
    output_cuda = llm_engine_cuda.gemm_quantized(input_tensor, weight_tensor, scales, m, n, k)

    # Reference execution in PyTorch
    w_dequant = weight_tensor.to(torch.float32) * scales
    output_ref = torch.matmul(input_tensor, w_dequant)

    assert torch.allclose(output_cuda, output_ref, atol=1e-3), "CUDA kernel output mismatches PyTorch reference"