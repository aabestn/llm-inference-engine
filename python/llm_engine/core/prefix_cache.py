import hashlib
from typing import Dict, List, Optional

class PrefixCache:
    """Implements prefix caching to reuse KV blocks across common prompt prefixes."""
    
    def __init__(self, block_size: int):
        self.block_size = block_size
        # Maps hash(prompt_prefix_tokens) -> physical block ID
        self.cache: Dict[str, int] = {}

    def _hash_tokens(self, token_ids: List[int]) -> str:
        return hashlib.sha256(bytes(token_ids)).hexdigest()

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Matches input tokens against existing prefix blocks to maximize cache reuse."""
        matched_blocks = []
        unmatched_tokens = token_ids
        
        # Check block-aligned prefix segments
        for i in range(0, len(token_ids), self.block_size):
            chunk = token_ids[i : i + self.block_size]
            if len(chunk) < self.block_size:
                break
            
            chunk_hash = self._hash_tokens(chunk)
            if chunk_hash in self.cache:
                matched_blocks.append(self.cache[chunk_hash])
                unmatched_tokens = token_ids[i + self.block_size :]
            else:
                break
                
        return matched_blocks, unmatched_tokens