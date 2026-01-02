"""
FastAPI server for distributed inference.
Provides HTTP endpoint for text generation using the naive inference pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# Import from naive-inference module
from naive_inference import load_model, generate


# Global variables for model and tokenizer
model = None
tokenizer = None


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The input prompt for text generation")
    max_new_tokens: int = Field(50, ge=1, le=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, gt=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - loads model on startup."""
    global model, tokenizer

    print("Starting up FastAPI server...")
    print("Loading model and tokenizer...")

    try:
        model, tokenizer = load_model()
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    yield

    # Cleanup on shutdown
    print("Shutting down server...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Distributed Inference API",
    description="API for text generation using naive inference pipeline",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt using the loaded model.

    Args:
        request: GenerateRequest containing prompt and generation parameters

    Returns:
        GenerateResponse with generated text and request parameters
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )

    try:
        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
