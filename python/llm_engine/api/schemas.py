from typing import List, Optional
from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for model generation")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    stream: bool = Field(default=True, description="Enable streaming token responses via Server-Sent Events")

class TokenStreamResponse(BaseModel):
    text: str = Field(..., description="Generated token chunk")
    finish_reason: Optional[str] = Field(default=None, description="Completion status (e.g., 'stop', 'length')")

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    finish_reason: str