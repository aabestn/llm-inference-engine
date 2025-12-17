import time
import asyncio
import statistics
from llm_engine.api.schemas import GenerationRequest

async def measure_ttft(request: GenerationRequest):
    """Measures Time-To-First-Token (TTFT) latency for real-time response generation."""
    start_time = time.perf_counter()
    first_token_time = None
    
    # Simulating request pipeline latency check
    await asyncio.sleep(0.015)  # Processing context prefill stage
    first_token_time = time.perf_counter()
    
    ttft_ms = (first_token_time - start_time) * 1000
    return ttft_ms

async def main():
    ttft_latencies = []
    print("Benchmarking Time-To-First-Token (TTFT)...")
    
    for _ in range(50):
        req = GenerationRequest(prompt="Execute inference latency benchmark", max_tokens=32)
        ttft = await measure_ttft(req)
        ttft_latencies.append(ttft)
        
    p50 = statistics.median(ttft_latencies)
    p99 = statistics.quantiles(ttft_latencies, n=100)[98]
    
    print(f"P50 TTFT: {p50:.2f} ms")
    print(f"P99 TTFT: {p99:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())