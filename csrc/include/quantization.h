#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <torch/extension.h>

/**
 * @brief FP8 / INT4 Quantized General Matrix Multiplication (GEMM) launch interface.
 * 
 * @param input Input activation tensor (scaled/quantized or FP16/BF16)
 * @param weight Quantized weight matrix (INT4 packed or FP8 format)
 * @param scales Scaling factors for activation/weight dequantization
 * @param output Output tensor storing unquantized/dequantized matrix result
 * @param m Row dimension of input
 * @param n Column dimension of weight matrix
 * @param k Inner contracting dimension
 */
torch::Tensor gemm_quantized(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& scales,
    int m,
    int n,
    int k
);

#endif // QUANTIZATION_H