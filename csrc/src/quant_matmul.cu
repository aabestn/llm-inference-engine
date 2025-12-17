#include "../include/quantization.h"
#include <cuda_runtime.h>

// CUDA Kernel execution for quantized Matrix Multiplication (INT4/FP8)
__global__ void gemm_quantized_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            // Unpack quantized weight, apply scale parameter, and accumulate GEMM dot product
            float act = input[row * k + i];
            float w_dequant = static_cast<float>(weight[i * n + col]) * scales[col];
            sum += act * w_dequant;
        }
        output[row * n + col] = sum;
    }
}

torch::Tensor gemm_quantized(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scales,
    int m, int n, int k
) {
    auto output = torch::zeros({m, n}, input.options());
    
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    gemm_quantized_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        output.data_ptr<float>(),
        m, n, k
    );

    return output;
}