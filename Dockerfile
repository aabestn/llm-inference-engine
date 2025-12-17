FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source tree and compile C++/CUDA extension modules
COPY . .
RUN pip3 install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "llm_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]