import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY", "secret-api-key") # Default for safety, should be set in .env
BASE_MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "../models/fine_tuned")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_api")

# Global variables for model storage
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    logger.info("Loading model and adapters...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Load Base Model
        # Quantization is disabled for 270M model as per pipeline recommendation for performance
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. Load LoRA Adapters
        if os.path.exists(ADAPTER_PATH):
            logger.info(f"Found adapters at {ADAPTER_PATH}, loading...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        else:
            logger.warning(f"Adapters not found at {ADAPTER_PATH}, using base model only.")
            model = base_model

        model.eval()
        
        model_artifacts["model"] = model
        model_artifacts["tokenizer"] = tokenizer
        model_artifacts["device"] = device
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Model loading failed")
        
    yield
    
    # Shutdown: Clear resources
    model_artifacts.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model resources released.")

app = FastAPI(title="EGX30 Banks QA API", version="1.0.0", lifespan=lifespan)

# CORS (Optional, good for web integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware for Monitoring ---
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log simplified metrics
    # In a real setup, this might push to Prometheus/Datadog
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s | "
        f"Throughput: {1/process_time:.2f} rps"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Auth Security ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# --- Data Models ---
class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in model_artifacts}

@app.post("/predict", response_model=AnswerResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: QuestionRequest):
    if "model" not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]
    
    start_gen = time.time()
    
    try:
        # Prompt Engineering (Match training format)
        # Note: In a RAG setup, context retrieval would happen here.
        # For this usage, we assume direct QA or prompt handling similar to chat.
        # Check if chat template is available or construct manually
        
        messages = [{"role": "user", "content": request.question}]
        
        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(device)
        
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "do_sample": True,
            "temperature": request.temperature,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)
            
        # Decode response (skip prompt)
        response_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        latency = (time.time() - start_gen) * 1000
        
        return AnswerResponse(
            answer=response_text.strip(),
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
