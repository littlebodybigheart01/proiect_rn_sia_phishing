import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

CONFIG_PATH = "config/preprocessing_config.yaml"


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fișierul de config nu există: {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def preprocess_and_split():
    print(f">> Încărcare configurare din {CONFIG_PATH}...")
    try:
        config = load_config(CONFIG_PATH)
    except Exception as e:
        print(f"[EROARE] Nu pot citi YAML: {e}")
        return

    input_path = config["dataset"]["input_path"]
    output_dir = config["dataset"].get("output_dir", "data")
    processed_dir = config["dataset"].get("processed_dir", os.path.join(output_dir, "processed"))
    processed_filename = config["dataset"].get("processed_filename", "processed.csv")

    print(f">> Citire dataset: {input_path}")
    if not os.path.exists(input_path):
        print(f"[EROARE FATALĂ] Fișierul {input_path} nu a fost găsit!")
        print("   Verifică dacă ai scris corect numele în config/preprocessing_config.yaml")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[EROARE] Nu pot citi CSV-ul: {e}")
        return

    # Curățare de bază
    stratify_col = config["split"].get("stratify_col", "label")
    if stratify_col not in df.columns:
        print(f"[EROARE] Coloana '{stratify_col}' lipsește din CSV!")
        return

    df = df.dropna(subset=["text", stratify_col])

    # Normalizări minimale pentru Etapa 3 (fără a distruge link-uri)
    text_series = df["text"].astype(str)
    if config.get("preprocessing", {}).get("lowercase", True):
        text_series = text_series.str.lower()
    text_series = text_series.str.replace(r"\s+", " ", regex=True).str.strip()

    min_len = int(config.get("preprocessing", {}).get("min_text_len", 6))
    df["text"] = text_series
    df = df[df["text"].str.len() >= min_len]

    if config.get("preprocessing", {}).get("remove_duplicates", True):
        df = df.drop_duplicates(subset=["text"])

    # Salvare dataset procesat (Etapa 3)
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, processed_filename)
    df.to_csv(processed_path, index=False)
    print(f"[OK] Dataset procesat salvat: {processed_path}")

    # Logică Split 70 / 15 / 15
    X = df["text"]
    y = df[stratify_col]
    train_size = config["split"]["train_size"]
    random_state = config["split"]["random_state"]

    # Pas 1: Train vs Rest
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    # Pas 2: Rest vs Val/Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # DataFrames
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df = pd.DataFrame({"text": X_val, "label": y_val})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    # Salvare
    paths = {
        "train": os.path.join(output_dir, "train", config["dataset"]["train_filename"]),
        "validation": os.path.join(output_dir, "validation", config["dataset"]["val_filename"]),
        "test": os.path.join(output_dir, "test", config["dataset"]["test_filename"]),
    }

    for key, path in paths.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if key == "train":
            train_df.to_csv(path, index=False)
        elif key == "validation":
            val_df.to_csv(path, index=False)
        elif key == "test":
            test_df.to_csv(path, index=False)

    print("\n[SUCCES] Dataset împărțit și salvat:")
    print(f" 1. Train ({len(train_df)}) -> {paths['train']}")
    print(f" 2. Val   ({len(val_df)})   -> {paths['validation']}")
    print(f" 3. Test  ({len(test_df)})  -> {paths['test']}")


if __name__ == "__main__":
    preprocess_and_split()
