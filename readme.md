# High-Performance LLM Inference & Serving Engine

An asynchronous LLM serving engine built using C++/CUDA custom kernels, continuous batching, and PagedAttention memory management.

## Key Features

* **PagedAttention**: Reduces VRAM fragmentation by dynamically allocating KV-cache blocks.
* **Quantization Support**: Native FP8 and INT4 GEMM kernels targeting NVIDIA Ampere and Ada Lovelace architectures.
* **Continuous Batching**: Dynamic request scheduling maximizing GPU compute utilization.
* **Prefix Caching**: Avoids redundant KV-cache computation across shared prompt prefixes.
* **Streaming API**: High-throughput FastAPI interface with low Time-To-First-Token (TTFT).

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/aabestn/llm-inference-engine.git
cd llm-inference-engine

# Build CUDA extensions and install in editable mode
pip install -e .

```

## Running the Server

```bash
uvicorn llm_engine.api.server:app --host 0.0.0.0 --port 8000

```