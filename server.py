"""
FastAPI server for distributed inference.
Provides HTTP endpoint for text generation using the inference engine.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

from distr_inference import InferenceEngine, print_metrics


# Global inference engine
engine: InferenceEngine | None = None


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int


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
    global engine

    print("Starting up FastAPI server...")
    print("Loading model...")

    try:
        engine = InferenceEngine()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    yield

    print("Shutting down server...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Distributed Inference API",
    description="API for text generation using the inference engine",
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
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )

    try:
        generated_text, metrics = engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )

        print_metrics(metrics)

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
        "model_loaded": engine is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
