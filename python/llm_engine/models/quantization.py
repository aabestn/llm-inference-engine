import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):
    """Linear layer wrapper executing FP8 / INT4 GEMM via custom CUDA extension kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_bits: int = 4,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_bits = quant_bits
        
        # Quantized weight and scale parameter tensors
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features // (8 // quant_bits)), dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "scales",
            torch.ones((out_features,), dtype=torch.float16, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import llm_engine_cuda
        
        m = x.shape[0]
        n = self.out_features
        k = self.in_features
        
        # Execute custom quantization matrix multiplication kernel
        return llm_engine_cuda.gemm_quantized(x, self.weight, self.scales, m, n, k)