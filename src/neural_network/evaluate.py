import os

# --- FIX CRITIC PENTRU COMPATIBILITATE TF 2.16 / Python 3.13 ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


# =========================
# CONFIG
# =========================
CONFIG_PATH = "config/preprocessing_config.yaml"

MODEL_DIR = "models/phishing_distilbert_multilingual"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

TEST_METRICS_JSON = os.path.join(RESULTS_DIR, "test_metrics.json")
CONF_MATRIX_PNG = os.path.join(DOCS_DIR, "confusion_matrix.png")

MAX_LENGTH = 128
BATCH_SIZE = 32


def load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu există config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_paths_from_config(cfg: dict):
    out_dir = cfg["dataset"].get("output_dir", "data")
    test_fn = cfg["dataset"]["test_filename"]
    test_path = os.path.join(out_dir, "test", test_fn)
    return test_path


def load_data_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc fișierul de test: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} trebuie să conțină coloanele: text, label")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, np.array(labels, dtype=np.int32)


def predict_probs_in_batches(model, tokenizer, texts):
    probs_all = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="tf",
        )
        out = model(enc, training=False)
        logits = out.logits
        probs = tf.nn.sigmoid(logits).numpy().flatten()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


def save_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitim", "Phishing"],
        yticklabels=["Legitim", "Phishing"],
    )
    plt.ylabel("Real")
    plt.xlabel("Prezis")
    plt.title("Matricea de Confuzie - DistilBERT")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    print(f">> Loading config: {CONFIG_PATH}")
    cfg = load_yaml(CONFIG_PATH)
    test_path = load_split_paths_from_config(cfg)

    print(">> Load test split")
    texts, y_true = load_data_csv(test_path)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Nu găsesc MODEL_DIR={MODEL_DIR}. Rulează train.py ca să genereze modelul."
        )

    print(">> Load tokenizer + model from MODEL_DIR")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    print(">> Predict (batched)")
    probs = predict_probs_in_batches(model, tokenizer, texts)
    y_pred = (probs >= 0.5).astype(int)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro"))
    test_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    test_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Legitim", "Phishing"],
        output_dict=True,
        zero_division=0,
    )

    print(">> Save confusion matrix")
    save_confusion_matrix(y_true, y_pred, CONF_MATRIX_PNG)
    print(f"[OK] Saved: {CONF_MATRIX_PNG}")

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "testaccuracy": test_acc,
        "testf1macro": test_f1,
        "testprecisionmacro": test_precision,
        "testrecallmacro": test_recall,
        "classificationreport": report,
        "artifacts": {
            "confusionmatrix_png": CONF_MATRIX_PNG,
            "model_dir": MODEL_DIR,
            "test_path": test_path,
        },
        "threshold": 0.5,
        "batch_size_eval": BATCH_SIZE,
        "max_length": MAX_LENGTH,
    }

    with open(TEST_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 macro: {test_f1:.4f}")
    print(f"Precision macro: {test_precision:.4f}")
    print(f"Recall macro: {test_recall:.4f}")
    print(f"[OK] Metrics saved: {TEST_METRICS_JSON}")


if __name__ == "__main__":
    main()
