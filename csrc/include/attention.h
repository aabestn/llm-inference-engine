#ifndef ATTENTION_H
#define ATTENTION_H

#include <torch/extension.h>

/**
 * @brief Performs PagedAttention v1 CUDA kernel execution.
 * 
 * @param out Output tensor for attention results [num_seqs, num_heads, head_size]
 * @param query Query tensor [num_seqs, num_heads, head_size]
 * @param key_cache Blocked Key Cache tensor [num_blocks, num_heads, head_size / x, block_size, x]
 * @param value_cache Blocked Value Cache tensor [num_blocks, num_heads, head_size, block_size]
 * @param block_tables Mapping table from logical sequence block index to physical block ID [num_seqs, max_blocks_per_seq]
 * @param context_lens Array storing current context lengths for each sequence [num_seqs]
 * @param max_context_len Maximum sequence context length in current batch
 * @param scale Softmax scaling factor (1.0 / sqrt(head_size))
 */
void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len,
    float scale
);

#endif // ATTENTION_H