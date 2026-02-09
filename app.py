import os

# --- 0. FIX CRITIC PENTRU COMPATIBILITATE (Trebuie să fie prima linie) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import time
from datetime import datetime

import streamlit as st
import tensorflow as tf
try:
    from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
except ImportError:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    DistilBertTokenizer = AutoTokenizer
    TFDistilBertForSequenceClassification = TFAutoModelForSequenceClassification
import numpy as np
from PIL import Image
import easyocr
import pandas as pd


# =========================
# 1) CONFIGURARE / PATH-URI
# =========================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DEFAULT_MODEL_PATH = "models/phishing_distilbert_multilingual"
OPTIMIZED_MODEL_DIR = "models/phishing_distilbert_multilingual_optimized"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

FEEDBACK_FILE = "data/feedback/user_feedback.csv"

FINAL_METRICS_PATH = "results/final_metrics.json"
OPT_EXPERIMENTS_PATH = "results/optimization_experiments.csv"
TEST_METRICS_PATH = "results/test_metrics.json"
TRAINING_HISTORY_PATH = "results/training_history.csv"
BASELINE_METRICS_PATH = "results/baseline_metrics.json"
TFLITE_LATENCY_PATH = "results/tflite_latency.json"

CONF_MAT_OPT_PATH = "docs/confusion_matrix_optimized.png"
CONF_MAT_BASE_PATH = "docs/confusion_matrix.png"
CONF_MAT_BASELINE_PATH = "docs/confusion_matrix_baseline.png"
LOSS_CURVE_PATH = "docs/loss_curve.png"

st.set_page_config(
    page_title="HAM OR SPAM PROTOCOL",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Session state (memoria aplicației)
defaults = {
    "last_text": None,
    "last_score": None,
    "last_latency_ms": None,
    "last_input_type": None,      # "text" / "ocr"
    "history": [],                # lista de evenimente inferență (runtime)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# 2) ICONS / TEXTS / CSS
# =========================

ICONS = {
    "skull": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff0033" stroke-width="2" stroke-linecap="square">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    """,
    "shield": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ff41" stroke-width="2" stroke-linecap="square">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <polyline points="9 12 11 14 15 10"/>
    </svg>
    """,
    "eye": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffff00" stroke-width="2" stroke-linecap="square">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
        <circle cx="12" cy="12" r="3"/>
    </svg>
    """,
    "chip": """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2" stroke-linecap="square">
        <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
        <line x1="9" y1="1" x2="9" y2="4"/>
        <line x1="15" y1="1" x2="15" y2="4"/>
        <line x1="9" y1="20" x2="9" y2="23"/>
        <line x1="20" y1="9" x2="23" y2="9"/>
        <line x1="1" y1="14" x2="4" y2="14"/>
    </svg>
    """,
}

TEXTS = {
    "ro": {
        "sidebar_title": "Setări Sistem",
        "tab_text": "TEXT INPUT",
        "tab_image": "IMG SCAN (OCR)",
        "tab_stats": "STATS",
        "input_label": ">> INTRODU FLUXUL DE DATE:",
        "upload_label": ">> UPLOAD SCREENSHOT:",
        "scan_btn": ">> INITIATE SCAN PROTOCOL <<",
        "ocr_success": ">> DATE EXTRASE:",
        "error_model": ">> EROARE SISTEM: MODELUL LIPSEȘTE.",
        "error_empty": ">> EROARE: NU EXISTĂ DATE.",
        "feedback_title": ">> VALIDARE HUMAN-IN-THE-LOOP:",
        "feedback_ok": "[ CORRECT ]",
        "feedback_bad": "[ WRONG ]",
        "feedback_success": ">> DATA LOGGED.",
        "stats_title": "STATISTICI ÎN TIMP REAL",
        "res_phishing": {"title": "CRITICAL THREAT DETECTED", "prob": "PROBABILITY:", "status": "STATUS: MALICIOUS"},
        "res_legit": {"title": "SYSTEM SECURE", "prob": "INTEGRITY:", "status": "STATUS: VERIFIED"},
        "res_suspect": {"title": "UNKNOWN SIGNATURE", "prob": "RISK:", "status": "STATUS: SUSPICIOUS"},
    },
    "en": {
        "sidebar_title": "System Config",
        "tab_text": "TEXT INPUT",
        "tab_image": "IMG SCAN (OCR)",
        "tab_stats": "STATS",
        "input_label": ">> DATA_STREAM_INPUT:",
        "upload_label": ">> UPLOAD SCREENSHOT:",
        "scan_btn": ">> INITIATE SCAN PROTOCOL <<",
        "ocr_success": ">> DATA_EXTRACTED:",
        "error_model": ">> SYSTEM ERROR: MODEL NOT FOUND.",
        "error_empty": ">> ERROR: NO DATA.",
        "feedback_title": ">> HUMAN-IN-THE-LOOP VALIDATION:",
        "feedback_ok": "[ CORRECT ]",
        "feedback_bad": "[ WRONG ]",
        "feedback_success": ">> DATA LOGGED.",
        "stats_title": "REALTIME STATS",
        "res_phishing": {"title": "CRITICAL THREAT DETECTED", "prob": "PROBABILITY:", "status": "STATUS: MALICIOUS"},
        "res_legit": {"title": "SYSTEM SECURE", "prob": "INTEGRITY:", "status": "STATUS: VERIFIED"},
        "res_suspect": {"title": "UNKNOWN SIGNATURE", "prob": "RISK:", "status": "STATUS: SUSPICIOUS"},
    },
}

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
    .stApp {
        background-color: #020202;
        background-image: linear-gradient(rgba(0, 255, 0, 0.02) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(0, 255, 0, 0.02) 1px, transparent 1px);
        background-size: 20px 20px;
        color: #00ff41;
        font-family: 'Share Tech Mono', monospace;
    }
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #ff00ff;
        text-align: center;
        text-shadow: 2px 2px 0px #00ffff;
        font-size: 45px !important;
        margin-bottom: 20px;
        letter-spacing: 3px;
    }
    .stTextArea textarea {
        background-color: #0a0a0a !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 18px !important;
        border-radius: 0px;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 10px #00ff00;
        border-color: #fff !important;
    }
    .stButton button {
        width: 100%;
        background: transparent;
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 20px;
        border: 2px solid #00ffff !important;
        border-radius: 0px;
        text-transform: uppercase;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: #00ffff;
        color: #000;
        box-shadow: 0 0 20px #00ffff;
    }
    .scan-box {
        border: 2px solid;
        padding: 20px;
        margin-top: 20px;
        background: #050505;
        display: flex;
        align-items: center;
        gap: 20px;
        font-family: 'Orbitron', sans-serif;
    }
    .box-phish { border-color: #ff0033; color: #ff0033; box-shadow: 0 0 15px rgba(255,0,51,0.3); }
    .box-safe  { border-color: #00ff41; color: #00ff41; box-shadow: 0 0 15px rgba(0,255,65,0.3); }
    .box-susp  { border-color: #ffff00; color: #ffff00; box-shadow: 0 0 15px rgba(255,255,0,0.3); }
    .res-title { font-size: 22px; font-weight: 900; margin-bottom: 5px; }
    .res-data { font-size: 18px; font-family: 'Share Tech Mono'; }
    .ai-console {
        border-left: 3px solid #00ffff;
        background: rgba(0, 255, 255, 0.05);
        padding: 15px;
        margin-top: 15px;
        color: #e0ffff;
        font-family: 'Share Tech Mono', monospace;
        display: flex;
        gap: 15px;
    }
    .feedback-title {
        margin-top: 30px;
        margin-bottom: 10px;
        text-align: center;
        color: #666;
        font-size: 14px;
        letter-spacing: 1px;
        border-top: 1px dashed #333;
        padding-top: 15px;
        font-family: 'Orbitron', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #000;
        color: #004400;
        border: 1px solid #004400;
        border-radius: 0px;
        font-family: 'Orbitron', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff00 !important;
        color: #000 !important;
        border: 1px solid #00ff00 !important;
    }
    .small-mono {
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        color: #88ff88;
        opacity: 0.85;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 3) FUNCȚII BACKEND
# =========================

def safe_load_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def safe_read_csv(path: str):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def resolve_model_path(default_path: str, optimized_path: str):
    # Prefer explicit env override if valid, then optimized, then default
    env_path = os.getenv("MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    if os.path.isdir(optimized_path):
        return optimized_path
    return default_path


MODEL_PATH = resolve_model_path(MODEL_PATH, OPTIMIZED_MODEL_DIR)


@st.cache_resource
def load_resources(model_path: str):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception:
        # fallback la modelul de baza daca optimized nu se poate incarca
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
            model = TFDistilBertForSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH)
            return tokenizer, model
        except Exception:
            return None, None


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["ro", "en"], gpu=False)


def predict_func(text: str, tokenizer, model):
    enc = tokenizer([text.lower()], truncation=True, padding=True, max_length=128, return_tensors="tf")
    t0 = time.perf_counter()
    out = model(enc)
    t1 = time.perf_counter()
    score = float(tf.nn.sigmoid(out.logits).numpy()[0][0])
    latency_ms = float((t1 - t0) * 1000.0)
    return score, latency_ms


def decide_bucket(score: float, th_low: float, th_high: float):
    if score >= th_high:
        return "phishing"
    if score <= th_low:
        return "legit"
    return "suspect"


def append_history(event: dict):
    st.session_state.history.append(event)
    # păstrăm doar ultimele 200 ca să nu crească infinit
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]


def compute_realtime_stats(history: list[dict]):
    if not history:
        return {
            "count": 0,
            "avg_score": None,
            "avg_latency": None,
            "p50_latency": None,
            "p90_latency": None,
            "bucket_counts": {},
        }

    df = pd.DataFrame(history)
    out = {"count": int(len(df))}

    scores = pd.to_numeric(df.get("score"), errors="coerce").dropna()
    lats = pd.to_numeric(df.get("latency_ms"), errors="coerce").dropna()

    out["avg_score"] = float(scores.mean()) if len(scores) else None
    out["avg_latency"] = float(lats.mean()) if len(lats) else None
    out["p50_latency"] = float(lats.median()) if len(lats) else None
    out["p90_latency"] = float(lats.quantile(0.9)) if len(lats) else None

    if "bucket" in df.columns:
        vc = df["bucket"].value_counts(dropna=False).to_dict()
        out["bucket_counts"] = {str(k): int(v) for k, v in vc.items()}
    else:
        out["bucket_counts"] = {}

    return out


def save_feedback(
    *,
    text: str,
    input_type: str,
    lang: str,
    model_path: str,
    score: float,
    latency_ms: float,
    pred_label: int,
    th_low: float,
    th_high: float,
    user_action: str,   # "correct" / "wrong"
    final_label: int,
):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "input_type": input_type,
        "lang": lang,
        "model_path": model_path,
        "score": float(score),
        "pred_label": int(pred_label),
        "threshold_low": float(th_low),
        "threshold_high": float(th_high),
        "latency_ms": float(latency_ms),
        "user_action": user_action,
        "final_label": int(final_label),
        "src": "user_feedback",
    }
    pd.DataFrame([row]).to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)


# =========================
# 4) SIDEBAR 
# =========================

with st.sidebar:
    st.markdown("### System")
    lang = "ro" if st.radio("LANG:", ["RO", "EN"], index=0) == "RO" else "en"
    T = TEXTS[lang]

    st.markdown(f"### {T['sidebar_title']}")

    st.caption("State Machine thresholds")
    th_low = st.slider("SAFE threshold (low)", 0.00, 0.49, 0.25, 0.01)
    th_high = st.slider("PHISHING threshold (high)", 0.51, 1.00, 0.75, 0.01)
    if th_low >= th_high:
        st.warning("threshold_low trebuie să fie < threshold_high")

    st.caption("Model")
    available_models = [
        ("Standard", DEFAULT_MODEL_PATH),
    ]
    if os.path.isdir(OPTIMIZED_MODEL_DIR):
        available_models.append(("Optimized", OPTIMIZED_MODEL_DIR))

    labels = [item[0] for item in available_models]
    paths = [item[1] for item in available_models]
    default_index = paths.index(MODEL_PATH) if MODEL_PATH in paths else 0
    choice = st.selectbox("Model selectat", labels, index=default_index)
    MODEL_PATH = paths[labels.index(choice)]

    st.code(MODEL_PATH, language="text")
    if os.path.basename(MODEL_PATH) == os.path.basename(OPTIMIZED_MODEL_DIR):
        st.success("Model activ: OPTIMIZED")
    else:
        st.info("Model activ: BASELINE")

    if st.button("Reset realtime stats (session)"):
        st.session_state.history = []
        st.session_state.last_text = None
        st.session_state.last_score = None
        st.session_state.last_latency_ms = None
        st.session_state.last_input_type = None


# =========================
# 5) UI: TITLU + TAB-URI
# =========================

st.markdown("<h1>IT'S HAM OR SPAM?</h1>", unsafe_allow_html=True)
st.caption("Realtime phishing detection for text and OCR input.")

tab1, tab2, tab3 = st.tabs([T["tab_text"], T["tab_image"], T["tab_stats"]])

current_input = ""
input_type = None

with tab1:
    txt = st.text_area(T["input_label"], height=100, key="txt_area")
    if txt and txt.strip():
        current_input = txt.strip()
        input_type = "text"

with tab2:
    up = st.file_uploader(T["upload_label"], type=["png", "jpg", "jpeg"])
    if up:
        img = Image.open(up)
        st.image(img, use_container_width=True)
        with st.spinner("OCR PROCESSING..."):
            try:
                res_list = load_ocr().readtext(np.array(img), detail=0)
                res = " ".join(res_list).strip()
                if res:
                    st.markdown(f"**{T['ocr_success']}**")
                    st.code(res)
                    current_input = res
                    input_type = "ocr"
            except Exception:
                st.error("OCR FAILED.")


# =========================
# 7) INFERENȚĂ + REZULTAT + REALTIME STATS + FEEDBACK
# =========================

tok, mod = load_resources(MODEL_PATH)
if tok is None or mod is None:
    st.error(T["error_model"])
else:
    if st.button(T["scan_btn"]):
        if current_input and current_input.strip():
            score, latency_ms = predict_func(current_input, tok, mod)

            st.session_state.last_text = current_input
            st.session_state.last_score = score
            st.session_state.last_latency_ms = latency_ms
            st.session_state.last_input_type = input_type or "text"

            # update realtime history
            bucket = decide_bucket(score, th_low, th_high)
            append_history(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "score": float(score),
                    "latency_ms": float(latency_ms),
                    "bucket": bucket,
                    "input_type": st.session_state.last_input_type,
                    "lang": lang,
                }
            )
        else:
            st.warning(T["error_empty"])

    # Realtime stats (se actualizează automat la fiecare scanare, Streamlit rerulează scriptul)
    rt = compute_realtime_stats(st.session_state.history)

    if st.session_state.last_text:
        sc = float(st.session_state.last_score or 0.0)
        pc = sc * 100.0
        lat_ms = float(st.session_state.last_latency_ms or 0.0)

        bucket = decide_bucket(sc, th_low, th_high)
        if bucket == "phishing":
            cat, css, ic = "res_phishing", "box-phish", ICONS["skull"]
        elif bucket == "legit":
            cat, css, ic = "res_legit", "box-safe", ICONS["shield"]
        else:
            cat, css, ic = "res_suspect", "box-susp", ICONS["eye"]

        st.markdown(
            f"""
                <div class="scan-box {css}">
                    <div style="min-width:40px">{ic}</div>
                    <div>
                        <div class="res-title">{T[cat]['title']}</div>
                        <div class="res-data">{T[cat]['prob']} {pc:.2f}% | {T[cat]['status']}</div>
                        <div class="small-mono">latency: {lat_ms:.2f} ms | session scans: {rt['count']}</div>
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.progress(min(max(sc, 0.0), 1.0))
        st.caption(f"score={sc:.4f} | thresholds low={th_low:.2f}, high={th_high:.2f} | model={MODEL_PATH}")

        # Realtime stats inline (după fiecare mesaj)
        with st.expander(T["stats_title"], expanded=True):
            a, b, c, d = st.columns(4)
            a.metric("Session scans", rt["count"])
            b.metric("Avg score", f"{rt['avg_score']:.3f}" if rt["avg_score"] is not None else "n/a")
            c.metric("Avg latency (ms)", f"{rt['avg_latency']:.2f}" if rt["avg_latency"] is not None else "n/a")
            d.metric("Latency p90 (ms)", f"{rt['p90_latency']:.2f}" if rt["p90_latency"] is not None else "n/a")

            st.write("Bucket counts:", rt["bucket_counts"])
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history).tail(10), use_container_width=True)

        # --- FEEDBACK LOOP ---
        st.markdown(f"<div class='feedback-title'>{T['feedback_title']}</div>", unsafe_allow_html=True)

        pred_label = 1 if sc >= 0.5 else 0

        c_left, c_center, c_right = st.columns([1, 2, 1])
        with c_center:
            if st.button(f"{T['feedback_ok']}", key="btn_ok", use_container_width=True):
                save_feedback(
                    text=st.session_state.last_text,
                    input_type=st.session_state.last_input_type,
                    lang=lang,
                    model_path=MODEL_PATH,
                    score=sc,
                    latency_ms=lat_ms,
                    pred_label=pred_label,
                    th_low=th_low,
                    th_high=th_high,
                    user_action="correct",
                    final_label=pred_label,
                )
                st.toast(T["feedback_success"])

            st.write("")

            if st.button(f"{T['feedback_bad']}", key="btn_bad", use_container_width=True):
                final_label = 1 - pred_label
                save_feedback(
                    text=st.session_state.last_text,
                    input_type=st.session_state.last_input_type,
                    lang=lang,
                    model_path=MODEL_PATH,
                    score=sc,
                    latency_ms=lat_ms,
                    pred_label=pred_label,
                    th_low=th_low,
                    th_high=th_high,
                    user_action="wrong",
                    final_label=final_label,
                )
                st.toast(T["feedback_success"])


# =========================
# 8) TAB STATS (global)
# =========================

with tab3:
    st.subheader(T["tab_stats"])

    # last inference
    if st.session_state.last_score is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Last score", f"{float(st.session_state.last_score):.4f}")
        c2.metric("Last latency (ms)", f"{float(st.session_state.last_latency_ms or 0.0):.2f}")
        c3.metric("Last input", st.session_state.last_input_type or "n/a")
    else:
        st.info("No inference yet in this session.")

    st.divider()

    # realtime (session) stats
    rt = compute_realtime_stats(st.session_state.history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session scans", rt["count"])
    c2.metric("Avg score", f"{rt['avg_score']:.3f}" if rt["avg_score"] is not None else "n/a")
    c3.metric("Latency p50 (ms)", f"{rt['p50_latency']:.2f}" if rt["p50_latency"] is not None else "n/a")
    c4.metric("Latency p90 (ms)", f"{rt['p90_latency']:.2f}" if rt["p90_latency"] is not None else "n/a")
    st.write("Bucket counts:", rt["bucket_counts"])

    with st.expander("Session history (last 25)"):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history).tail(25), use_container_width=True)
        else:
            st.write("Empty.")

    st.divider()

    # Dataset stats (Etapa 3/5)
    split_paths = {
        "train": "data/train/train.csv",
        "validation": "data/validation/validation.csv",
        "test": "data/test/test.csv",
    }
    split_stats = []
    for k, path in split_paths.items():
        df = safe_read_csv(path)
        if df is not None and not df.empty and "label" in df.columns:
            split_stats.append(
                {
                    "split": k,
                    "rows": len(df),
                    "label_0": int((df["label"] == 0).sum()),
                    "label_1": int((df["label"] == 1).sum()),
                }
            )

    if split_stats:
        st.markdown("### Dataset splits")
        st.dataframe(pd.DataFrame(split_stats), use_container_width=True)
    else:
        st.caption("Dataset splits not found (expected data/train|validation|test/*.csv).")

    st.divider()

    # Etapa 5 metrics / plots (dacă există)
    tm = safe_load_json(TEST_METRICS_PATH)
    if tm:
        st.markdown("### results/test_metrics.json")
        st.json(tm)
    else:
        st.caption("results/test_metrics.json not found.")

    if os.path.exists(CONF_MAT_BASE_PATH):
        st.caption("docs/confusion_matrix.png")
        st.image(CONF_MAT_BASE_PATH, use_container_width=True)
    else:
        st.caption("docs/confusion_matrix.png not found.")

    st.divider()

    # Baseline model metrics (bonus)
    bm = safe_load_json(BASELINE_METRICS_PATH)
    if bm:
        st.markdown("### results/baseline_metrics.json")
        st.json(bm)
    else:
        st.caption("results/baseline_metrics.json not found.")

    if os.path.exists(CONF_MAT_BASELINE_PATH):
        st.caption("docs/confusion_matrix_baseline.png")
        st.image(CONF_MAT_BASELINE_PATH, use_container_width=True)
    else:
        st.caption("docs/confusion_matrix_baseline.png not found.")

    st.divider()

    # TFLite latency (bonus)
    lt = safe_load_json(TFLITE_LATENCY_PATH)
    if lt:
        st.markdown("### results/tflite_latency.json")
        st.json(lt)
    else:
        st.caption("results/tflite_latency.json not found.")

    st.divider()

    # Etapa 6 metrics / plots (dacă există)
    fm = safe_load_json(FINAL_METRICS_PATH)
    if fm:
        st.markdown("### results/final_metrics.json")
        st.json(fm)
    else:
        st.caption("results/final_metrics.json not found.")

    colA, colB = st.columns(2)
    with colA:
        if os.path.exists(CONF_MAT_OPT_PATH):
            st.caption("docs/confusion_matrix_optimized.png")
            st.image(CONF_MAT_OPT_PATH, use_container_width=True)
        else:
            st.caption("docs/confusion_matrix_optimized.png not found.")
    with colB:
        if os.path.exists(LOSS_CURVE_PATH):
            st.caption("docs/loss_curve.png")
            st.image(LOSS_CURVE_PATH, use_container_width=True)
        else:
            st.caption("docs/loss_curve.png not found.")

    st.divider()

    # Etapa 6 extra plots (dacă există)
    opt_acc = "docs/optimization/accuracy_comparison.png"
    opt_f1 = "docs/optimization/f1_comparison.png"
    opt_curves = "docs/optimization/learning_curves_best.png"
    res_curves = "docs/results/learning_curves_final.png"
    res_metrics = "docs/results/metrics_evolution.png"
    res_examples = "docs/results/example_predictions.png"

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(opt_acc):
            st.caption(opt_acc)
            st.image(opt_acc, use_container_width=True)
        else:
            st.caption(f"{opt_acc} not found.")
    with c2:
        if os.path.exists(opt_f1):
            st.caption(opt_f1)
            st.image(opt_f1, use_container_width=True)
        else:
            st.caption(f"{opt_f1} not found.")

    c3, c4 = st.columns(2)
    with c3:
        if os.path.exists(opt_curves):
            st.caption(opt_curves)
            st.image(opt_curves, use_container_width=True)
        else:
            st.caption(f"{opt_curves} not found.")
    with c4:
        if os.path.exists(res_curves):
            st.caption(res_curves)
            st.image(res_curves, use_container_width=True)
        else:
            st.caption(f"{res_curves} not found.")

    c5, c6 = st.columns(2)
    with c5:
        if os.path.exists(res_metrics):
            st.caption(res_metrics)
            st.image(res_metrics, use_container_width=True)
        else:
            st.caption(f"{res_metrics} not found.")
    with c6:
        if os.path.exists(res_examples):
            st.caption(res_examples)
            st.image(res_examples, use_container_width=True)
        else:
            st.caption(f"{res_examples} not found.")

    st.divider()

    # Training history (CSV)
    th = safe_read_csv(TRAINING_HISTORY_PATH)
    if th is not None and not th.empty:
        st.markdown("### Training history")
        st.dataframe(th.tail(25), use_container_width=True)
    else:
        st.caption("results/training_history.csv not found.")

    # feedback file stats (persistent)
    fb = safe_read_csv(FEEDBACK_FILE)
    st.markdown("### Feedback log (persistent)")
    if fb is None or fb.empty:
        st.caption("No feedback yet.")
    else:
        st.caption(f"Rows: {len(fb)}")
        with st.expander("Last 25 feedback rows"):
            st.dataframe(fb.tail(25), use_container_width=True)

    exp = safe_read_csv(OPT_EXPERIMENTS_PATH)
    if exp is not None and not exp.empty:
        st.markdown("### Optimization experiments")
        st.dataframe(exp, use_container_width=True)
