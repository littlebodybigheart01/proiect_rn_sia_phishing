import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf


def load_sample_texts(path: str, num_samples: int) -> list[str]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])
    texts = df["text"].astype(str).tolist()
    if not texts:
        raise ValueError("No texts available for benchmarking.")
    return texts[:num_samples]


def load_vectorizer_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def vectorize_texts(texts: list[str], meta: dict) -> np.ndarray:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=int(meta["max_tokens"]),
        output_sequence_length=int(meta["seq_len"]),
        standardize=meta.get("standardize", "lower_and_strip_punctuation"),
        split=meta.get("split", "whitespace"),
    )
    vectorizer.set_vocabulary(meta["vocabulary"])
    seqs = vectorizer(tf.constant(texts))
    return tf.cast(seqs, tf.int32).numpy()


def convert_to_tflite(model_dir: str, output_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)


def benchmark_tflite(model_path: str, sequences: np.ndarray, warmup: int, runs: int) -> dict:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    try:
        total_samples = sequences.shape[0]
        for i in range(warmup):
            seq = sequences[i % total_samples]
            interpreter.set_tensor(input_index, seq[None, :])
            interpreter.invoke()
            _ = interpreter.get_tensor(output_index)

        start = time.perf_counter()
        for i in range(runs):
            seq = sequences[i % total_samples]
            interpreter.set_tensor(input_index, seq[None, :])
            interpreter.invoke()
            _ = interpreter.get_tensor(output_index)
        total = time.perf_counter() - start

        return {
            "avg_latency_ms": float((total / runs) * 1000.0),
            "runs": int(runs),
            "warmup_runs": int(warmup),
        }
    except RuntimeError as exc:
        return {
            "avg_latency_ms": None,
            "runs": int(runs),
            "warmup_runs": int(warmup),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline model to TFLite and benchmark latency.")
    parser.add_argument("--model_dir", default="models/baseline_text_model_infer")
    parser.add_argument("--output_path", default="models/final_model.tflite")
    parser.add_argument("--test_csv", default="data/test/test.csv")
    parser.add_argument("--vectorizer_json", default="models/baseline_text_vectorizer.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--latency_json", default="results/tflite_latency.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.latency_json), exist_ok=True)

    convert_to_tflite(args.model_dir, args.output_path)
    print(f"[OK] TFLite model saved: {args.output_path}")

    texts = load_sample_texts(args.test_csv, args.num_samples)
    meta = load_vectorizer_meta(args.vectorizer_json)
    sequences = vectorize_texts(texts, meta)
    latency = benchmark_tflite(args.output_path, sequences, args.warmup, args.runs)
    latency["model_path"] = args.output_path
    latency["model_dir"] = args.model_dir
    latency["vectorizer_json"] = args.vectorizer_json

    with open(args.latency_json, "w", encoding="utf-8") as f:
        json.dump(latency, f, indent=2)

    print(f"[OK] Latency results saved: {args.latency_json}")


if __name__ == "__main__":
    main()
