import argparse
import json
import os
import time

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def make_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["text"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(vectorizer: tf.keras.layers.TextVectorization, embedding_dim: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        vectorizer.vocabulary_size(),
        embedding_dim,
        name="embedding",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_embed")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_hidden")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_inference_model(trained_model: tf.keras.Model, seq_len: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_ids")
    x = trained_model.get_layer("embedding")(inputs)
    x = trained_model.get_layer("avg_pool")(x)
    x = trained_model.get_layer("dropout_embed")(x)
    x = trained_model.get_layer("dense_hidden")(x)
    x = trained_model.get_layer("dropout_hidden")(x)
    outputs = trained_model.get_layer("output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def save_confusion_matrix(cm, output_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - Baseline")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Phishing"])
    ax.set_yticklabels(["Legit", "Phishing"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight baseline text model.")
    parser.add_argument("--train", default="data/train/train.csv")
    parser.add_argument("--val", default="data/validation/validation.csv")
    parser.add_argument("--test", default="data/test/test.csv")
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model_dir", default="models/baseline_text_model")
    parser.add_argument("--tflite_model_dir", default="models/baseline_text_model_infer")
    parser.add_argument("--keras_path", default="models/baseline_text_model.keras")
    parser.add_argument("--metrics_path", default="results/baseline_metrics.json")
    parser.add_argument("--confusion_path", default="docs/confusion_matrix_baseline.png")
    parser.add_argument("--vectorizer_json", default="models/baseline_text_vectorizer.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.confusion_path), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tflite_model_dir, exist_ok=True)

    train_df = load_csv(args.train)
    val_df = load_csv(args.val)
    test_df = load_csv(args.test)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=args.max_tokens,
        output_sequence_length=args.seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(train_df["text"].values)

    train_ds = make_dataset(train_df, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_df, args.batch_size, shuffle=False)
    test_ds = make_dataset(test_df, args.batch_size, shuffle=False)

    model = build_model(vectorizer, args.embedding_dim, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    start_time = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_time_sec = time.perf_counter() - start_time

    y_true = test_df["label"].values
    y_prob = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "train_time_sec": float(train_time_sec),
        "artifacts": {
            "model_dir": args.model_dir,
            "keras_path": args.keras_path,
            "tflite_model_dir": args.tflite_model_dir,
            "vectorizer_json": args.vectorizer_json,
            "confusion_matrix_png": args.confusion_path,
            "metrics_json": args.metrics_path,
        },
    }

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, args.confusion_path)

    # SavedModel export for TFLite (text input), plus native Keras format for reuse.
    model.export(args.model_dir)
    model.save(args.keras_path, include_optimizer=False)

    # Export inference-only SavedModel (int input) for TFLite without string ops.
    infer_model = build_inference_model(model, args.seq_len)
    infer_model.export(args.tflite_model_dir)

    vocab = vectorizer.get_vocabulary()
    vectorizer_meta = {
        "max_tokens": args.max_tokens,
        "seq_len": args.seq_len,
        "standardize": "lower_and_strip_punctuation",
        "split": "whitespace",
        "vocabulary": vocab,
    }
    with open(args.vectorizer_json, "w", encoding="utf-8") as f:
        json.dump(vectorizer_meta, f, indent=2)

    print(f"[OK] Baseline SavedModel exported: {args.model_dir}")
    print(f"[OK] Baseline Keras saved: {args.keras_path}")
    print(f"[OK] Baseline inference SavedModel exported: {args.tflite_model_dir}")
    print(f"[OK] Vectorizer saved: {args.vectorizer_json}")
    print(f"[OK] Metrics saved: {args.metrics_path}")
    print(f"[OK] Confusion matrix saved: {args.confusion_path}")


if __name__ == "__main__":
    main()
