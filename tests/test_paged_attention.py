import pytest
import torch

def test_paged_attention_v1():
    """Validates output shape and execution bounds for PagedAttention CUDA kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    import llm_engine_cuda

    num_seqs = 2
    num_heads = 8
    head_size = 64
    block_size = 16
    max_blocks = 4

    out = torch.zeros((num_seqs, num_heads, head_size), dtype=torch.float32, device="cuda")
    query = torch.randn((num_seqs, num_heads, head_size), dtype=torch.float32, device="cuda")
    key_cache = torch.randn((10, num_heads, head_size // 8, block_size, 8), dtype=torch.float32, device="cuda")
    value_cache = torch.randn((10, num_heads, head_size, block_size), dtype=torch.float32, device="cuda")
    block_tables = torch.tensor([[0, 1, -1, -1], [2, 3, 4, -1]], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([32, 48], dtype=torch.int32, device="cuda")

    scale = 1.0 / (head_size ** 0.5)

    llm_engine_cuda.paged_attention_v1(
        out, query, key_cache, value_cache, block_tables, context_lens, 48, scale
    )

    assert out.shape == (num_seqs, num_heads, head_size)
    assert not torch.isnan(out).any(), "PagedAttention kernel produced NaN values"