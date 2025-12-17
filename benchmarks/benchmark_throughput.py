import time
import torch
from llm_engine.core.scheduler import ContinuousBatchScheduler, SequenceRequest
from llm_engine.core.cache_engine import CacheEngine

def run_throughput_benchmark(
    num_requests: int = 100,
    input_len: int = 512,
    output_len: int = 128,
    batch_size: int = 32
):
    """Measures token generation throughput across continuous batch sizes."""
    print(f"Starting benchmark: {num_requests} requests | Input: {input_len} | Output: {output_len}")
    
    cache_engine = CacheEngine(num_blocks=1024, block_size=16, num_heads=32, head_size=128)
    scheduler = ContinuousBatchScheduler(cache_engine=cache_engine, max_batch_size=batch_size)
    
    # Pre-populate request queue
    for i in range(num_requests):
        prompt_tokens = [100] * input_len
        req = SequenceRequest(request_id=f"req_{i}", prompt_tokens=prompt_tokens, max_tokens=output_len)
        scheduler.add_request(req)

    start_time = time.perf_counter()
    total_tokens_generated = 0

    # Simulate generation engine processing loop
    while scheduler.waiting_queue or scheduler.running_queue:
        active_batch = scheduler.schedule()
        if not active_batch:
            break
            
        # Simulate forward step execution
        for req in active_batch:
            req.generated_tokens.append(200)
            total_tokens_generated += 1
            if len(req.generated_tokens) >= req.max_tokens:
                scheduler.running_queue.remove(req)
                cache_engine.free(req.block_table)

    elapsed_time = time.perf_counter() - start_time
    tokens_per_sec = total_tokens_generated / elapsed_time
    
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/second")

if __name__ == "__main__":
    run_throughput_benchmark()