import os
import time
import uuid
from typing import Optional, Tuple

os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Keep API stable for browser extension by defaulting to CPU inference.
# Set API_USE_GPU=1 only if you explicitly want GPU for this process.
if os.getenv("API_USE_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
try:
    from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
except ImportError:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    DistilBertTokenizer = AutoTokenizer
    TFDistilBertForSequenceClassification = TFAutoModelForSequenceClassification

DEFAULT_MODEL_PATH = "models/phishing_distilbert_multilingual"
OPTIMIZED_MODEL_DIR = "models/phishing_distilbert_multilingual_optimized"
SCAN_FILE = "data/feedback/extension_scans.csv"
FEEDBACK_FILE = "data/feedback/extension_feedback.csv"
MAX_INPUT_CHARS = 5000


def resolve_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    if os.path.isdir(OPTIMIZED_MODEL_DIR):
        return OPTIMIZED_MODEL_DIR
    return DEFAULT_MODEL_PATH


def normalize_choice(choice: Optional[str]) -> str:
    value = (choice or "auto").strip().lower()
    aliases = {
        "baseline": "standard",
        "default": "standard",
        "normal": "standard",
        "optimised": "optimized",
        "opt": "optimized",
    }
    return aliases.get(value, value)


def load_model(path: str):
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = TFDistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model


MODEL_CACHE: dict[str, Tuple[object, object]] = {}


def get_model(path: str):
    if path not in MODEL_CACHE:
        # Keep only one loaded model in memory to avoid OOM in long sessions.
        if MODEL_CACHE:
            MODEL_CACHE.clear()
            tf.keras.backend.clear_session()
        MODEL_CACHE[path] = load_model(path)
    return MODEL_CACHE[path]


def resolve_choice(choice: Optional[str]) -> str:
    normalized = normalize_choice(choice)
    if normalized == "auto":
        return resolve_model_path()
    if normalized == "standard":
        if os.path.isdir(DEFAULT_MODEL_PATH):
            return DEFAULT_MODEL_PATH
        fallback = resolve_model_path()
        if os.path.isdir(fallback):
            return fallback
        raise HTTPException(status_code=400, detail="Standard model not available")
    if normalized == "optimized":
        if os.path.isdir(OPTIMIZED_MODEL_DIR):
            return OPTIMIZED_MODEL_DIR
        if os.path.isdir(DEFAULT_MODEL_PATH):
            return DEFAULT_MODEL_PATH
        fallback = resolve_model_path()
        if os.path.isdir(fallback):
            return fallback
        raise HTTPException(status_code=400, detail="Optimized model not available")
    raise HTTPException(status_code=400, detail="Invalid model choice")


app = FastAPI(title="Phishing Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold_low: Optional[float] = 0.25
    threshold_high: Optional[float] = 0.75
    model: Optional[str] = "auto"  # standard | optimized | auto


class FeedbackRequest(BaseModel):
    scan_id: str
    user_action: str = Field(..., pattern="^(correct|wrong)$")
    label: Optional[str] = None
    score: Optional[float] = None
    latency_ms: Optional[float] = None


def predict(text: str, model_path: str) -> tuple[float, float]:
    tokenizer, model = get_model(model_path)
    enc = tokenizer([text.lower()], truncation=True, padding=True, max_length=128, return_tensors="tf")
    t0 = time.perf_counter()
    out = model(enc)
    t1 = time.perf_counter()
    score = float(tf.nn.sigmoid(out.logits).numpy()[0][0])
    latency_ms = float((t1 - t0) * 1000.0)
    return score, latency_ms


def bucket(score: float, low: float, high: float) -> str:
    if score >= high:
        return "PHISH"
    if score <= low:
        return "SAFE"
    return "SUSPECT"


def log_scan(scan_id: str, text: str, label: str, score: float, latency_ms: float, model_path: str):
    os.makedirs(os.path.dirname(SCAN_FILE), exist_ok=True)
    header = "ts,scan_id,text,label,score,latency_ms,model_path\n"
    row = f"{time.time():.0f},{scan_id},\"{text.replace('"','""')}\",{label},{score:.6f},{latency_ms:.2f},{model_path}\n"
    if not os.path.exists(SCAN_FILE):
        with open(SCAN_FILE, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(row)
    else:
        with open(SCAN_FILE, "a", encoding="utf-8") as f:
            f.write(row)


def log_feedback(scan_id: str, user_action: str, label: Optional[str], score: Optional[float], latency_ms: Optional[float]):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    header = "ts,scan_id,user_action,label,score,latency_ms\n"
    row = f"{time.time():.0f},{scan_id},{user_action},{label or ''},{score or ''},{latency_ms or ''}\n"
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(row)
    else:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(row)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": resolve_model_path(),
        "api_use_gpu": os.getenv("API_USE_GPU", "0") == "1",
        "available_models": {
            "standard": os.path.isdir(DEFAULT_MODEL_PATH),
            "optimized": os.path.isdir(OPTIMIZED_MODEL_DIR),
        },
        "model_paths": {
            "standard": DEFAULT_MODEL_PATH,
            "optimized": OPTIMIZED_MODEL_DIR,
        },
    }


@app.post("/scan")
def scan(req: ScanRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    low = float(req.threshold_low or 0.25)
    high = float(req.threshold_high or 0.75)
    if low >= high:
        raise HTTPException(status_code=400, detail="threshold_low must be < threshold_high")

    normalized_model = normalize_choice(req.model)
    model_path = resolve_choice(normalized_model)

    try:
        score, latency_ms = predict(text, model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error: {exc}")

    label = bucket(score, low, high)
    scan_id = uuid.uuid4().hex
    log_scan(scan_id, text, label, score, latency_ms, model_path)
    return {
        "scan_id": scan_id,
        "score": score,
        "label": label,
        "latency_ms": latency_ms,
        "model_choice": normalized_model,
        "model_path": model_path,
        "truncated": len(text) == MAX_INPUT_CHARS,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not req.scan_id:
        raise HTTPException(status_code=400, detail="scan_id missing")
    log_feedback(req.scan_id, req.user_action, req.label, req.score, req.latency_ms)
    return {"status": "ok"}
