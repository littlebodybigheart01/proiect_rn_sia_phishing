import os
import shutil

import numpy as np
import pandas as pd

# --- CONFIGURARE CĂI ---
# Asigură-te că numele fișierelor de aici corespund exact cu ce ai în folderul data/raw/
FILE_EMAIL = "data/raw/emailreal.csv"
FILE_SMS = "data/raw/smsreal.csv"
FILE_AI_RO = "data/raw/phishing_ai_ro_only.csv"
FILE_AI_RO_GENERATED = "data/generated/phishing_ai_ro_only.csv"
FILE_AI_PATCH = "data/raw/phishing_ai_targeted_patch.csv"

OUTPUT_FILE = "data/raw/multilingualdataset.csv"


def clean_text(text):
    """
    Funcție optimizată pentru curățarea textului.
    Elimină caracterele care strică structura CSV (Newlines, Tabs).
    """
    if not isinstance(text, str):
        return ""
    # Înlocuim Enter și Tab cu spațiu simplu
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Eliminăm spațiile multiple (ex: "  " -> " ")
    return " ".join(text.split())


def process_email_dataset():
    """Procesează Phishing_Email.csv"""
    if not os.path.exists(FILE_EMAIL):
        print(f"[SKIP] Nu am găsit fișierul: {FILE_EMAIL}")
        return pd.DataFrame()

    print(f"[1/3] Procesare Email-uri ({FILE_EMAIL})...")
    try:
        # Citim doar coloanele relevante pentru a economisi memorie
        # 'Email Text' și 'Email Type' sunt numele standard din acest dataset
        df = pd.read_csv(FILE_EMAIL, usecols=['Email Text', 'Email Type'])

        # Redenumire standard
        df.rename(columns={'Email Text': 'text', 'Email Type': 'label_raw'}, inplace=True)

        # Mapare Etichete: Safe Email -> 0, Phishing Email -> 1
        # Folosim .strip() pentru a elimina spațiile ascunse din etichete
        df['label'] = df['label_raw'].astype(str).str.strip().map({
            'Safe Email': 0,
            'Phishing Email': 1
        })

        # Adăugăm metadate
        df['type'] = 'email'
        df['source'] = 'real_enron'
        df['lang'] = 'en'

        # Eliminăm rândurile care nu au putut fi etichetate (erori in CSV original)
        df.dropna(subset=['label'], inplace=True)

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea Email: {e}")
        return pd.DataFrame()


def process_sms_dataset():
    """Procesează spam.csv (SMS Collection)"""
    if not os.path.exists(FILE_SMS):
        print(f"[SKIP] Nu am găsit fișierul: {FILE_SMS}")
        return pd.DataFrame()

    print(f"[2/3] Procesare SMS-uri ({FILE_SMS})...")
    try:
        # Acest fișier are encoding 'latin-1' adesea
        # Coloanele sunt v1 (label) și v2 (text)
        df = pd.read_csv(FILE_SMS, encoding='latin-1', usecols=['v1', 'v2'])

        # Redenumire
        df.rename(columns={'v2': 'text', 'v1': 'label_raw'}, inplace=True)

        # Mapare: ham -> 0, spam -> 1
        df['label'] = df['label_raw'].map({'ham': 0, 'spam': 1})

        df['type'] = 'sms'
        df['source'] = 'real_sms_collection'
        df['lang'] = 'en'

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea SMS: {e}")
        return pd.DataFrame()


def _sync_generated_copy(source_path: str, generated_path: str):
    if os.path.exists(source_path) and not os.path.exists(generated_path):
        os.makedirs(os.path.dirname(generated_path), exist_ok=True)
        try:
            shutil.copyfile(source_path, generated_path)
        except Exception as exc:
            print(f"[WARN] Nu pot copia in {generated_path}: {exc}")


def process_ai_ro_dataset():
    """Procesează datele generate de AI (Română)"""
    file_path = None
    if os.path.exists(FILE_AI_RO_GENERATED):
        file_path = FILE_AI_RO_GENERATED
    elif os.path.exists(FILE_AI_RO):
        file_path = FILE_AI_RO

    if not file_path:
        print(f"[SKIP] Nu am găsit datele AI ({FILE_AI_RO_GENERATED} / {FILE_AI_RO})...")
        return pd.DataFrame()

    print(f"[3/3] Procesare Date AI Română ({file_path})...")
    try:
        df = pd.read_csv(file_path)

        if "text" not in df.columns or "label" not in df.columns:
            print("[SKIP] Format invalid pentru AI RO (lipsește text/label).")
            return pd.DataFrame()

        # Asigurăm compatibilitatea coloanelor
        if 'lang' not in df.columns: df['lang'] = 'ro'
        if 'source' not in df.columns: df['source'] = 'ai_generated'
        if "type" not in df.columns: df["type"] = "mixed"

        # AI-ul generează deja 0/1, deci nu trebuie mapare, doar verificare
        # Dacă coloanele au alte nume în fișierul generat, le poți redenumi aici

        if file_path == FILE_AI_RO:
            _sync_generated_copy(FILE_AI_RO, FILE_AI_RO_GENERATED)

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea AI RO: {e}")
        return pd.DataFrame()


def process_ai_patch_dataset():
    """Procesează date suplimentare AI (patch/targeted) dacă există"""
    if not os.path.exists(FILE_AI_PATCH):
        print(f"[SKIP] Nu am găsit datele AI patch ({FILE_AI_PATCH})...")
        return pd.DataFrame()

    print(f"[4/4] Procesare Date AI Patch ({FILE_AI_PATCH})...")
    try:
        df = pd.read_csv(FILE_AI_PATCH)
        if "label" not in df.columns or "text" not in df.columns:
            print("[SKIP] Format invalid pentru AI patch (lipsește text/label).")
            return pd.DataFrame()

        if "lang" not in df.columns:
            df["lang"] = "ro"
        if "source" not in df.columns:
            df["source"] = "ai_patch"
        if "type" not in df.columns:
            df["type"] = "mixed"

        return df[["text", "label", "type", "source", "lang"]]
    except Exception as e:
        print(f"[EROARE] La procesarea AI patch: {e}")
        return pd.DataFrame()


def main():
    print("==========================================")
    print("   UNIFICARE DATASET-URI PENTRU PHISHING  ")
    print("==========================================")

    # 1. Încărcare
    df_email = process_email_dataset()
    df_sms = process_sms_dataset()
    df_ai = process_ai_ro_dataset()
    df_ai_patch = process_ai_patch_dataset()

    # 2. Combinare
    # Folosind o listă și pd.concat este metoda cea mai eficientă de memorie
    datasets = [df_email, df_sms, df_ai, df_ai_patch]

    # Filtrare dataset-uri goale
    valid_datasets = [d for d in datasets if not d.empty]

    if not valid_datasets:
        print("[EROARE CRITICĂ] Niciun set de date nu a putut fi încărcat!")
        return

    df_final = pd.concat(valid_datasets, ignore_index=True)
    print(f"\n[INFO] Total rânduri brute: {len(df_final)}")

    # 3. Curățare Avansată
    print("[INFO] Se curăță textul și se elimină duplicatele...")

    # Eliminare rânduri fără text sau label
    df_final.dropna(subset=['text', 'label'], inplace=True)

    # Curățare text (eliminare newlines pentru a nu strica CSV-ul final)
    df_final['text'] = df_final['text'].astype(str).apply(clean_text)

    # Eliminare duplicate (foarte important când combini mai multe surse)
    initial_len = len(df_final)
    df_final.drop_duplicates(subset=['text'], inplace=True)
    print(f"   -> Duplicate eliminate: {initial_len - len(df_final)}")

    # Eliminare texte prea scurte (ex: "ok", "hi") care nu sunt relevante
    df_final = df_final[df_final['text'].str.len() > 5]

    # Convertire label la int (pentru siguranță)
    df_final['label'] = df_final['label'].astype(int)

    # 4. Amestecare (Shuffle)
    # Este vital să amestecăm datele ca să nu avem toate cele în română la final
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Salvare
    # quoting=1 pune ghilimele în jurul textului, protejând virgulele din interiorul mesajelor
    df_final.to_csv(OUTPUT_FILE, index=False, quoting=1)

    print("\n" + "=" * 40)
    print("   STATISTICI FINALE")
    print("=" * 40)
    print(f"Fișier salvat: {OUTPUT_FILE}")
    print(f"Total Mesaje:  {len(df_final)}")
    print("-" * 20)
    print("Distribuție Clase:")
    print(df_final['label'].value_counts().to_string())
    print("-" * 20)
    print("Distribuție Limbi:")
    print(df_final['lang'].value_counts().to_string())
    print("=" * 40)

    # Salveaza statistici utile pentru documentatie (Etapa 4)
    try:
        os.makedirs("docs", exist_ok=True)
        text_len = df_final["text"].astype(str).str.len()
        stats_rows = [
            {"metric": "total_samples", "value": int(len(df_final))},
            {"metric": "label_0_count", "value": int((df_final["label"] == 0).sum())},
            {"metric": "label_1_count", "value": int((df_final["label"] == 1).sum())},
            {"metric": "avg_text_len", "value": float(text_len.mean())},
            {"metric": "median_text_len", "value": float(text_len.median())},
            {"metric": "p95_text_len", "value": float(text_len.quantile(0.95))},
        ]
        for lang, count in df_final["lang"].value_counts().items():
            stats_rows.append({"metric": f"lang_{lang}_count", "value": int(count)})
        for source, count in df_final["source"].value_counts().items():
            stats_rows.append({"metric": f"source_{source}_count", "value": int(count)})

        pd.DataFrame(stats_rows).to_csv("docs/data_statistics.csv", index=False)
        print("[OK] Statistici salvate: docs/data_statistics.csv")
    except Exception as exc:
        print(f"[WARN] Nu pot salva data_statistics.csv: {exc}")


if __name__ == "__main__":
    main()
