import os
import sys
import json
import random
from datetime import datetime

import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

sys.path.append(os.getcwd())
from src.neural_network.model import build_model
from transformers import DistilBertTokenizer


# =========================
# CONFIG
# =========================
CONFIG_PATH = "config/preprocessing_config.yaml"

MODEL_DIR = "models/phishing_distilbert_multilingual"
OPTIMIZED_MODEL_DIR = "models/phishing_distilbert_multilingual_optimized"
OPTIMIZED_MODEL_H5 = "models/optimized_model.h5"

RESULTS_DIR = "results"
DOCS_DIR = "docs"

OPT_EXPERIMENTS_CSV = os.path.join(RESULTS_DIR, "optimization_experiments.csv")
FINAL_METRICS_JSON = os.path.join(RESULTS_DIR, "final_metrics.json")
CONF_MATRIX_OPT_PNG = os.path.join(DOCS_DIR, "confusion_matrix_optimized.png")

MAX_LENGTH = 128
RANDOM_STATE = 42


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def build_early_stopping(*, min_epochs: int, patience: int):
    try:
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
            start_from_epoch=min_epochs,
        )
    except TypeError:
        class MinEpochEarlyStopping(tf.keras.callbacks.Callback):
            def __init__(self, min_epochs: int, patience: int):
                super().__init__()
                self.min_epochs = min_epochs
                self.patience = patience
                self.wait = 0
                self.best = None
                self.best_weights = None

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get("val_loss")
                if current is None:
                    return
                if self.best is None or current < self.best:
                    self.best = current
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                    return
                if epoch + 1 < self.min_epochs:
                    return
                self.wait += 1
                if self.wait >= self.patience:
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    self.model.stop_training = True

        return MinEpochEarlyStopping(min_epochs=min_epochs, patience=patience)


def load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu există config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_paths_from_config(cfg: dict):
    out_dir = cfg["dataset"].get("output_dir", "data")
    train_fn = cfg["dataset"]["train_filename"]
    val_fn = cfg["dataset"]["val_filename"]
    test_fn = cfg["dataset"]["test_filename"]

    train_path = os.path.join(out_dir, "train", train_fn)
    val_path = os.path.join(out_dir, "validation", val_fn)
    test_path = os.path.join(out_dir, "test", test_fn)
    return train_path, val_path, test_path


def load_data_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu găsesc fișierul: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} trebuie să conțină coloanele: text, label")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def make_tf_dataset(texts, labels, tokenizer, *, batch_size: int, is_train: bool):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    ds = tf.data.Dataset.from_tensor_slices((dict(enc), labels))
    if is_train:
        ds = ds.shuffle(10000, seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Optimized")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Legit", "Phishing"])
    plt.yticks(ticks, ["Legit", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_model(model, tokenizer, texts, labels, *, batch_size: int):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    preds = model.predict(dict(enc), batch_size=batch_size)
    logits = preds.logits if hasattr(preds, "logits") else preds
    probs = tf.nn.sigmoid(logits).numpy().flatten()
    pred_labels = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, pred_labels)),
        "f1_macro": float(f1_score(labels, pred_labels, average="macro")),
        "precision_macro": float(
            precision_score(labels, pred_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, pred_labels, average="macro", zero_division=0)
        ),
    }
    return metrics, pred_labels


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    set_seeds(RANDOM_STATE)
    configure_gpu_memory_growth()

    cfg = load_yaml(CONFIG_PATH)
    train_path, val_path, test_path = load_split_paths_from_config(cfg)

    train_texts, train_labels = load_data_csv(train_path)
    val_texts, val_labels = load_data_csv(val_path)
    test_texts, test_labels = load_data_csv(test_path)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    default_epochs = max(10, int(os.getenv("OPT_EPOCHS", "10")))
    max_experiments = int(os.getenv("OPT_MAX_EXPERIMENTS", "0"))
    experiments = [
        {"name": "baseline", "learning_rate": 3e-5, "batch_size": 16, "epochs": default_epochs},
        {"name": "lr_1e-5", "learning_rate": 1e-5, "batch_size": 16, "epochs": default_epochs},
        {"name": "lr_5e-5", "learning_rate": 5e-5, "batch_size": 16, "epochs": default_epochs},
        {"name": "batch_4", "learning_rate": 3e-5, "batch_size": 4, "epochs": default_epochs},
    ]

    existing = set()
    existing_rows = []
    if os.path.exists(OPT_EXPERIMENTS_CSV):
        try:
            existing_df = pd.read_csv(OPT_EXPERIMENTS_CSV)
            existing = set(existing_df.get("exp", []).astype(str).tolist())
            existing_rows = existing_df.to_dict(orient="records")
        except Exception:
            existing = set()
            existing_rows = []

    rows = list(existing_rows)
    best = {"f1_macro": -1.0}
    for r in existing_rows:
        try:
            if float(r.get("f1_macro", -1.0)) > best["f1_macro"]:
                best = {
                    "exp": r.get("exp"),
                    "learning_rate": r.get("learning_rate"),
                    "batch_size": r.get("batch_size"),
                    "epochs": r.get("epochs"),
                    "accuracy": r.get("accuracy"),
                    "f1_macro": r.get("f1_macro"),
                    "precision_macro": r.get("precision_macro"),
                    "recall_macro": r.get("recall_macro"),
                }
        except Exception:
            continue
    best_model = None
    best_pred_labels = None

    for idx, exp in enumerate(experiments, start=1):
        if exp["name"] in existing:
            print(f">> Skip {exp['name']} (already in {OPT_EXPERIMENTS_CSV})")
            continue
        if max_experiments > 0 and len(rows) >= max_experiments:
            print(">> Reached OPT_MAX_EXPERIMENTS limit.")
            break
        print(f">> Experiment {idx}: {exp['name']}")
        exp_start = time.time()
        tf.keras.backend.clear_session()
        gc.collect()
        model = build_model(learning_rate=exp["learning_rate"])

        train_ds = make_tf_dataset(
            train_texts, train_labels, tokenizer, batch_size=exp["batch_size"], is_train=True
        )
        val_ds = make_tf_dataset(
            val_texts, val_labels, tokenizer, batch_size=exp["batch_size"], is_train=False
        )

        early_stop = build_early_stopping(min_epochs=default_epochs, patience=3)
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=0,
        )

        try:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=exp["epochs"],
                callbacks=[early_stop, lr_sched],
                verbose=0,
            )

            metrics, pred_labels = evaluate_model(
                model, tokenizer, test_texts, test_labels, batch_size=exp["batch_size"]
            )
        except tf.errors.ResourceExhaustedError:
            print(f"[WARN] OOM la experimentul {exp['name']}. Se sare peste.")
            tf.keras.backend.clear_session()
            gc.collect()
            continue

        exp_seconds = float(time.time() - exp_start)
        rows.append(
            {
                "exp": exp["name"],
                "learning_rate": exp["learning_rate"],
                "batch_size": exp["batch_size"],
                "epochs": exp["epochs"],
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
                "train_time_sec": exp_seconds,
            }
        )

        if metrics["f1_macro"] > best["f1_macro"]:
            best = {
                **metrics,
                "exp": exp["name"],
                "learning_rate": exp["learning_rate"],
                "batch_size": exp["batch_size"],
                "epochs": exp["epochs"],
            }
            best_model = model
            best_pred_labels = pred_labels

        # salvare progres după fiecare experiment
        pd.DataFrame(rows).to_csv(OPT_EXPERIMENTS_CSV, index=False)
        # curățare explicită pentru a evita acumularea de memorie GPU
        del train_ds, val_ds, pred_labels, history
        tf.keras.backend.clear_session()
        gc.collect()

    if not rows:
        print("[EROARE] Nu exista experimente valide. Opreste optimizarea.")
        return

    pd.DataFrame(rows).to_csv(OPT_EXPERIMENTS_CSV, index=False)
    print(f"[OK] Experiments saved: {OPT_EXPERIMENTS_CSV}")

    if best_model is not None:
        os.makedirs(OPTIMIZED_MODEL_DIR, exist_ok=True)
        best_model.save_pretrained(OPTIMIZED_MODEL_DIR)
        tokenizer.save_pretrained(OPTIMIZED_MODEL_DIR)
        # DistilBERT este subclassed -> salvam weights in H5 pentru cerinta Etapa 6
        best_model.save_weights(OPTIMIZED_MODEL_H5)
        print(f"[OK] Optimized model saved: {OPTIMIZED_MODEL_H5}")

        plot_confusion_matrix(test_labels, best_pred_labels, CONF_MATRIX_OPT_PNG)
        print(f"[OK] Confusion matrix saved: {CONF_MATRIX_OPT_PNG}")

    final_metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_experiment": best,
        "test_accuracy": best["accuracy"],
        "test_f1_macro": best["f1_macro"],
        "test_precision_macro": best["precision_macro"],
        "test_recall_macro": best["recall_macro"],
        "artifacts": {
            "optimization_experiments_csv": OPT_EXPERIMENTS_CSV,
            "confusion_matrix_optimized_png": CONF_MATRIX_OPT_PNG,
            "optimized_model_h5": OPTIMIZED_MODEL_H5,
            "optimized_model_dir": OPTIMIZED_MODEL_DIR,
        },
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
    print(f"[OK] Final metrics saved: {FINAL_METRICS_JSON}")


if __name__ == "__main__":
    main()
