from typing import List, Deque
from collections import deque
from .cache_engine import CacheEngine

class SequenceRequest:
    def __init__(self, request_id: str, prompt_tokens: List[int], max_tokens: int):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.generated_tokens: List[int] = []
        self.max_tokens = max_tokens
        self.block_table: List[int] = []

class ContinuousBatchScheduler:
    """Schedules pending and running requests using continuous batching techniques."""
    
    def __init__(self, cache_engine: CacheEngine, max_batch_size: int = 32):
        self.cache_engine = cache_engine
        self.max_batch_size = max_batch_size
        self.waiting_queue: Deque[SequenceRequest] = deque()
        self.running_queue: List[SequenceRequest] = []

    def add_request(self, request: SequenceRequest) -> None:
        self.waiting_queue.append(request)

    def schedule(self) -> List[SequenceRequest]:
        """Dynamically forms batches from waiting and active running sequences."""
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size:
            req = self.waiting_queue.popleft()
            # Attempt physical cache block allocation for initial context step
            req.block_table = self.cache_engine.allocate(1)
            self.running_queue.append(req)
            
        return self.running_queue