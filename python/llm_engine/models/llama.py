import torch
import torch.nn as nn
from typing import Optional
from .quantization import QuantizedLinear

class LlamaAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention layer using PagedAttention execution."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Quantized projection layers
        self.q_proj = QuantizedLinear(hidden_size, num_heads * self.head_dim)
        self.k_proj = QuantizedLinear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = QuantizedLinear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = QuantizedLinear(num_heads * self.head_dim, hidden_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor
    ) -> torch.Tensor:
        import llm_engine_cuda

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Dispatch execution to custom PagedAttention CUDA kernel
        out = torch.empty_like(q)
        scale = 1.0 / (self.head_dim ** 0.5)
        
        llm_engine_cuda.paged_attention_v1(
            out, q, key_cache, value_cache, block_tables, context_lens, context_lens.max().item(), scale
        )
        
        return self.o_proj(out)


class LlamaForCausalLM(nn.Module):
    """Simplified Llama architecture utilizing custom quantized layers and PagedAttention kernels."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads)
            for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = QuantizedLinear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        key_caches: torch.Tensor,
        value_caches: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(positions, x, key_caches[i], value_caches[i], block_tables, context_lens)
        return self.lm_head(x)