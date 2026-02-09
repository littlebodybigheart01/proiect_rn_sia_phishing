import os
import sys

# --- FIX CRITIC PENTRU COMPATIBILITATE (Trebuie să fie prima linie) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from transformers import DistilBertTokenizer
sys.path.append(os.getcwd())
from src.neural_network.model import build_model


# =========================
# CONFIG
# =========================

CONFIG_PATH = "config/preprocessing_config.yaml"

MODEL_DIR = "models/phishing_distilbert_multilingual"
TRAINED_MODEL_H5 = "models/trained_model.h5"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

TRAINING_HISTORY_CSV = os.path.join(RESULTS_DIR, "training_history.csv")
TEST_METRICS_JSON = os.path.join(RESULTS_DIR, "test_metrics.json")

LOSS_CURVE_PNG = os.path.join(DOCS_DIR, "loss_curve.png")
CONF_MATRIX_PNG = os.path.join(DOCS_DIR, "confusion_matrix.png")

CHECKPOINT_DIR = os.path.join("models", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.weights.h5")

# Plafon epoci: recomandat 50 (cu EarlyStopping care oprește automat) [README Etapa 5]
EPOCHS = 50
MIN_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_LENGTH = 128

RANDOM_STATE = 42


@dataclass
class RunConfig:
    model_dir: str
    results_dir: str
    docs_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    random_state: int


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
            verbose=1,
            start_from_epoch=min_epochs,
        )
    except TypeError:
        # fallback pentru versiuni fara start_from_epoch
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
        raise FileNotFoundError(
            f"Nu găsesc fișierul: {path}. Rulează mai întâi preprocess_and_split.py!"
        )
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} trebuie să conțină coloanele: text, label")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def make_tf_dataset(texts, labels, tokenizer, *, is_train: bool):
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

    # cache + prefetch pentru viteză
    ds = ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def plot_training_curves(history: tf.keras.callbacks.History, out_path: str):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_training_curves_from_df(df: pd.DataFrame, out_path: str):
    if df.empty:
        return
    epochs = df["epoch"] + 1 if "epoch" in df.columns else np.arange(1, len(df) + 1)
    acc = df.get("accuracy", pd.Series([]))
    val_acc = df.get("val_accuracy", pd.Series([]))
    loss = df.get("loss", pd.Series([]))
    val_loss = df.get("val_loss", pd.Series([]))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_history_df(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
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


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    set_seeds(RANDOM_STATE)
    configure_gpu_memory_growth()

    print(f">> Loading config: {CONFIG_PATH}")
    cfg = load_yaml(CONFIG_PATH)

    train_path, val_path, test_path = load_split_paths_from_config(cfg)

    print(">> 1) Load data splits")
    train_texts, train_labels = load_data_csv(train_path)
    val_texts, val_labels = load_data_csv(val_path)
    test_texts, test_labels = load_data_csv(test_path)

    print(">> 2) Tokenizer")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"   [OK] Tokenizer saved to {MODEL_DIR}")

    train_ds = make_tf_dataset(train_texts, train_labels, tokenizer, is_train=True)
    val_ds = make_tf_dataset(val_texts, val_labels, tokenizer, is_train=False)

    # Callbacks (conform README: early stopping când val_loss nu scade 5 epoci + LR scheduler + CSVLogger)
    early_stop = build_early_stopping(min_epochs=MIN_EPOCHS, patience=5)

    history_df = load_history_df(TRAINING_HISTORY_CSV)
    initial_epoch = 0
    resume = False
    if history_df is not None:
        initial_epoch = int(len(history_df))
        if os.path.exists(CHECKPOINT_PATH):
            resume = initial_epoch > 0
        else:
            print("[WARN] training_history.csv exista dar checkpoint lipseste. Reincep antrenarea.")
            initial_epoch = 0
            resume = False

    callbacks = [
        early_stop,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(TRAINING_HISTORY_CSV, append=resume),
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),
    ]

    print(f">> 3) Train model (max {EPOCHS} epochs, early stopping enabled)")
    model = build_model(learning_rate=LEARNING_RATE)

    if resume:
        model.load_weights(CHECKPOINT_PATH)
        print(f"[OK] Resume din checkpoint: {CHECKPOINT_PATH} (epoca {initial_epoch})")

    history = None
    if initial_epoch >= EPOCHS:
        print(">> Training deja finalizat, se sare peste fit().")
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
        )

    # best epoch estimat din val_loss (util pentru raport / README)
    history_df = load_history_df(TRAINING_HISTORY_CSV)
    best_epoch = None
    epochs_ran = 0
    if history_df is not None and "val_loss" in history_df.columns:
        best_epoch = int(history_df["val_loss"].idxmin() + 1)
        epochs_ran = int(len(history_df))
    elif history is not None:
        val_losses = history.history.get("val_loss", [])
        best_epoch = int(np.argmin(val_losses) + 1) if len(val_losses) else None
        epochs_ran = int(len(history.history.get("loss", [])))

    print(">> 4) Save model (HuggingFace + H5)")
    model.save_pretrained(MODEL_DIR)
    print(f"   [OK] Model saved to {MODEL_DIR}")
    # DistilBERT este subclassed -> salvam weights in H5 pentru cerinta Etapa 5
    model.save_weights(TRAINED_MODEL_H5)
    print(f"   [OK] Model weights saved to {TRAINED_MODEL_H5}")

    print(">> 5) Save loss/accuracy curve")
    if history_df is not None:
        plot_training_curves_from_df(history_df, LOSS_CURVE_PNG)
    elif history is not None:
        plot_training_curves(history, LOSS_CURVE_PNG)
    print(f"   [OK] Saved: {LOSS_CURVE_PNG}")

    print(">> 6) Evaluate on test set")
    test_enc = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )
    preds = model.predict(dict(test_enc), batch_size=BATCH_SIZE)
    logits = preds.logits if hasattr(preds, "logits") else preds
    probs = tf.nn.sigmoid(logits).numpy().flatten()
    pred_labels = (probs >= 0.5).astype(int)

    test_acc = float(accuracy_score(test_labels, pred_labels))
    test_f1 = float(f1_score(test_labels, pred_labels, average="macro"))
    test_precision = float(precision_score(test_labels, pred_labels, average="macro", zero_division=0))
    test_recall = float(recall_score(test_labels, pred_labels, average="macro", zero_division=0))

    report = classification_report(
        test_labels,
        pred_labels,
        target_names=["Legit", "Phishing"],
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrix(test_labels, pred_labels, CONF_MATRIX_PNG)

    run_cfg = RunConfig(
        model_dir=MODEL_DIR,
        results_dir=RESULTS_DIR,
        docs_dir=DOCS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        random_state=RANDOM_STATE,
    )

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "classification_report": report,
        "best_epoch": best_epoch,
        "stopped_epoch": int(getattr(early_stop, "stopped_epoch", 0)),
        "epochs_ran": int(epochs_ran),
        "artifacts": {
            "traininghistory_csv": TRAINING_HISTORY_CSV,
            "testmetrics_json": TEST_METRICS_JSON,
            "loss_curve_png": LOSS_CURVE_PNG,
            "confusion_matrix_png": CONF_MATRIX_PNG,
            "model_dir": MODEL_DIR,
            "trained_model_h5": TRAINED_MODEL_H5,
        },
        "run_config": asdict(run_cfg),
    }

    with open(TEST_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 macro: {test_f1:.4f}")
    print(f"Precision macro: {test_precision:.4f}")
    print(f"Recall macro: {test_recall:.4f}")
    print(f"Best epoch (val_loss): {best_epoch}")
    print(f"[OK] Metrics saved: {TEST_METRICS_JSON}")
    print(f"[OK] Confusion matrix saved: {CONF_MATRIX_PNG}")
