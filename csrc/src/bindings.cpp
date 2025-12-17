#include <torch/extension.h>
#include "../include/attention.h"
#include "../include/quantization.h"

// Pybind11 module bindings exposing C++/CUDA extensions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-Performance LLM Inference & Serving Engine CUDA Kernels";
    
    m.def("paged_attention_v1", &paged_attention_v1, "PagedAttention v1 CUDA implementation");
    m.def("gemm_quantized", &gemm_quantized, "Quantized FP8/INT4 GEMM CUDA implementation");
}