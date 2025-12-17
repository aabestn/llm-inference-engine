import torch
from typing import List, Tuple

class CacheEngine:
    """Manages physical Key-Value Cache memory blocks allocated across GPU memory."""
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Pre-allocate key and value physical block cache memory on GPU
        self.key_cache = torch.empty(
            (num_blocks, num_heads, head_size // 8, block_size, 8),
            dtype=dtype,
            device=device
        )
        self.value_cache = torch.empty(
            (num_blocks, num_heads, head_size, block_size),
            dtype=dtype,
            device=device
        )
        self.free_blocks = list(range(num_blocks))

    def allocate(self, num_required_blocks: int) -> List[int]:
        """Allocates physical blocks for active sequence requests."""
        if len(self.free_blocks) < num_required_blocks:
            raise MemoryError("Out of KV Cache memory blocks")
        
        allocated = self.free_blocks[:num_required_blocks]
        self.free_blocks = self.free_blocks[num_required_blocks:]
        return allocated

    def free(self, block_ids: List[int]) -> None:
        """Frees blocks back into available pool upon request completion."""
        self.free_blocks.extend(block_ids)