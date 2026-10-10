"""FastAPI service for MiniGPT.

Run locally:   uvicorn api:app --host 0.0.0.0 --port 8000
Endpoints:     GET /  |  GET /health  |  POST /generate
Environment:   CHECKPOINT_PATH (default: minigpt_best.pt next to this file)
               CHECKPOINT_URL  (optional: downloaded at startup if the checkpoint file is missing)
"""
import os
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from minigpt import count_parameters, generate_text, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(HERE, "minigpt_best.pt"))
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL")

STATE = {}
LOCK = threading.Lock()


def _ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    if not CHECKPOINT_URL:
        raise RuntimeError(f"checkpoint not found at {CHECKPOINT_PATH} and CHECKPOINT_URL is not set")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_checkpoint()
    model, tokenizer, _ = load_model(CHECKPOINT_PATH, device="cpu")
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    yield
    STATE.clear()


app = FastAPI(title="MiniGPT API", version="1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=500)
    max_new_tokens: int = Field(200, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy decoding")
    top_k: Optional[int] = Field(None, ge=1, le=100)
    seed: Optional[int] = Field(None, description="set for reproducible output")


@app.get("/")
def root():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    model = STATE["model"]
    return {
        "name": "MiniGPT (from-scratch Transformer, character-level, Tiny Shakespeare)",
        "parameters": count_parameters(model),
        "context_length": model.cfg.context_length,
        "vocab_size": model.cfg.vocab_size,
        "usage": "POST /generate with JSON {prompt, max_new_tokens, temperature, top_k, seed}",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/generate")
def generate(req: GenerateRequest):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    model, tokenizer = STATE["model"], STATE["tokenizer"]
    ignored = sorted(set(req.prompt) - set(tokenizer.chars))
    start = time.time()
    with LOCK:
        if req.seed is not None:
            torch.manual_seed(req.seed)
        generated = generate_text(model, tokenizer, req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    return {
        "prompt": req.prompt,
        "generated": generated,
        "ignored_characters": ignored,
        "parameters": count_parameters(model),
        "context_length": model.cfg.context_length,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "seconds": round(time.time() - start, 3),
    }
