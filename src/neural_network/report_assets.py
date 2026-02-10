import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

CONFIG_PATH = "config/preprocessing_config.yaml"
RESULTS_DIR = "results"
DOCS_DIR = "docs"
OPT_DIR = os.path.join(DOCS_DIR, "optimization")
RES_DIR = os.path.join(DOCS_DIR, "results")

OPT_EXPERIMENTS_CSV = os.path.join(RESULTS_DIR, "optimization_experiments.csv")
TRAINING_HISTORY_CSV = os.path.join(RESULTS_DIR, "training_history.csv")
TEST_METRICS_JSON = os.path.join(RESULTS_DIR, "test_metrics.json")
FINAL_METRICS_JSON = os.path.join(RESULTS_DIR, "final_metrics.json")

CONFUSION_OPT = os.path.join(DOCS_DIR, "confusion_matrix_optimized.png")

MODEL_DIR_OPT = "models/phishing_distilbert_multilingual_optimized"
MODEL_DIR_BASE = "models/phishing_distilbert_multilingual"

MAX_LENGTH = 128


def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    os.makedirs(OPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)


def has_model_files(path: str) -> bool:
    return os.path.exists(os.path.join(path, "config.json")) and (
        os.path.exists(os.path.join(path, "tf_model.h5"))
    )


def resolve_model_dir() -> str | None:
    if os.path.isdir(MODEL_DIR_OPT) and has_model_files(MODEL_DIR_OPT):
        return MODEL_DIR_OPT
    if os.path.isdir(MODEL_DIR_BASE) and has_model_files(MODEL_DIR_BASE):
        return MODEL_DIR_BASE
    return None


def load_test_split() -> pd.DataFrame | None:
    cfg = load_yaml(CONFIG_PATH)
    test_fn = cfg["dataset"]["test_filename"]
    test_path = os.path.join(cfg["dataset"].get("output_dir", "data"), "test", test_fn)
    if not os.path.exists(test_path):
        print("[SKIP] No test split found.")
        return None
    df = pd.read_csv(test_path)
    if df.empty:
        print("[SKIP] Test split empty.")
        return None
    if "text" not in df.columns or "label" not in df.columns:
        print("[SKIP] Test split missing text/label.")
        return None
    return df


def load_model_bundle():
    model_dir = resolve_model_dir()
    if not model_dir:
        print("[SKIP] No model dir found.")
        return None, None, None

    tokenizer_dir = model_dir
    if not os.path.exists(os.path.join(tokenizer_dir, "vocab.txt")):
        tokenizer_dir = MODEL_DIR_BASE if os.path.isdir(MODEL_DIR_BASE) else "distilbert-base-multilingual-cased"

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_dir)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
    return model_dir, tokenizer, model


def plot_experiment_metrics():
    if not os.path.exists(OPT_EXPERIMENTS_CSV):
        print("[SKIP] No optimization_experiments.csv found.")
        return
    df = pd.read_csv(OPT_EXPERIMENTS_CSV)
    if df.empty:
        print("[SKIP] optimization_experiments.csv is empty.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(df["exp"], df["accuracy"], marker="o")
    plt.title("Accuracy per Experiment")
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "accuracy_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["exp"], df["f1_macro"], marker="o", color="#ff6600")
    plt.title("F1 Macro per Experiment")
    plt.xlabel("Experiment")
    plt.ylabel("F1 Macro")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "f1_comparison.png"))
    plt.close()


def plot_learning_curves():
    if not os.path.exists(TRAINING_HISTORY_CSV):
        print("[SKIP] No training_history.csv found.")
        return
    df = pd.read_csv(TRAINING_HISTORY_CSV)
    if df.empty:
        print("[SKIP] training_history.csv is empty.")
        return

    epochs = df["epoch"] + 1
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, df["accuracy"], label="Train Acc")
    plt.plot(epochs, df["val_accuracy"], label="Val Acc")
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OPT_DIR, "learning_curves_best.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, df["loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.title("Learning Curves (Final)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "learning_curves_final.png"))
    plt.close()


def plot_metrics_evolution():
    if not os.path.exists(TEST_METRICS_JSON) or not os.path.exists(FINAL_METRICS_JSON):
        print("[SKIP] No test_metrics.json or final_metrics.json found.")
        return

    with open(TEST_METRICS_JSON, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(FINAL_METRICS_JSON, "r", encoding="utf-8") as f:
        final = json.load(f)

    labels = ["Etapa 5", "Etapa 6"]
    acc = [base.get("test_accuracy"), final.get("test_accuracy")]
    f1 = [base.get("test_f1_macro"), final.get("test_f1_macro")]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, acc, width, label="Accuracy")
    plt.bar(x + width / 2, f1, width, label="F1 Macro")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.title("Metrics Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "metrics_evolution.png"))
    plt.close()


def plot_example_predictions(df, tokenizer, model):
    if df is None:
        return
    sample = df.sample(n=min(9, len(df)), random_state=42).reset_index(drop=True)

    enc = tokenizer(
        sample["text"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    preds = model.predict(dict(enc), batch_size=4)
    logits = preds.logits if hasattr(preds, "logits") else preds
    probs = tf.nn.sigmoid(logits).numpy().flatten()
    pred_labels = (probs >= 0.5).astype(int)

    plt.figure(figsize=(12, 8))
    plt.axis("off")
    for i in range(len(sample)):
        text = sample.loc[i, "text"]
        true_lbl = int(sample.loc[i, "label"])
        pred_lbl = int(pred_labels[i])
        prob = float(probs[i])
        short = (text[:120] + "...") if len(text) > 120 else text
        line = f"[{i+1}] true={true_lbl} pred={pred_lbl} p={prob:.2f} :: {short}"
        plt.text(0.01, 0.95 - i * 0.1, line, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "example_predictions.png"))
    plt.close()


def save_misclassified_examples(df, tokenizer, model, *, max_examples: int = 5):
    if df is None:
        return
    enc = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    preds = model.predict(dict(enc), batch_size=8)
    logits = preds.logits if hasattr(preds, "logits") else preds
    probs = tf.nn.sigmoid(logits).numpy().flatten()
    pred_labels = (probs >= 0.5).astype(int)

    true_labels = df["label"].astype(int).to_numpy()
    wrong_idx = np.where(pred_labels != true_labels)[0]
    if len(wrong_idx) == 0:
        print("[SKIP] No misclassified examples.")
        return

    confidence = np.where(pred_labels == 1, probs, 1 - probs)
    wrong_df = pd.DataFrame(
        {
            "index": wrong_idx,
            "true_label": true_labels[wrong_idx],
            "pred_label": pred_labels[wrong_idx],
            "confidence": confidence[wrong_idx],
            "text": df.iloc[wrong_idx]["text"].astype(str).values,
        }
    )
    wrong_df = wrong_df.sort_values("confidence", ascending=False).head(max_examples)
    wrong_df["text"] = wrong_df["text"].str.slice(0, 240)
    wrong_df.to_csv(os.path.join(RES_DIR, "misclassified_examples.csv"), index=False)
    print("[OK] Misclassified examples saved: docs/results/misclassified_examples.csv")


def copy_confusion_matrix():
    if os.path.exists(CONFUSION_OPT):
        target = os.path.join(RES_DIR, "confusion_matrix_optimized.png")
        if CONFUSION_OPT != target:
            import shutil
            shutil.copyfile(CONFUSION_OPT, target)


def main():
    ensure_dirs()
    plot_experiment_metrics()
    plot_learning_curves()
    plot_metrics_evolution()
    test_df = load_test_split()
    _, tokenizer, model = load_model_bundle()
    if tokenizer is not None and model is not None:
        plot_example_predictions(test_df, tokenizer, model)
        save_misclassified_examples(test_df, tokenizer, model)
    copy_confusion_matrix()
    print("[OK] Report assets generated.")


if __name__ == "__main__":
    main()
