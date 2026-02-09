import os
import time
import json
import random
import re
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURARE ---
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
TARGET_COUNT = int(os.getenv("AI_TARGET_COUNT", "15000"))
BATCH_SIZE = 25
OUTPUT_FILE = os.path.join("data", "raw", "phishing_ai_ro_only.csv")
OUTPUT_FILE_GENERATED = os.path.join("data", "generated", "phishing_ai_ro_only.csv")
MODEL_NAME = "gemini-flash-latest"

TOPICS_RO = {
    "Bancar": [
        "actualizare date personale (Banca Transilvania/BCR/ING/Raiffeisen)",
        "plată suspectă către comerciant extern",
        "card blocat temporar - necesită deblocare",
        "cod de autorizare tranzacție (3D Secure)",
        "cont accesat de pe un dispozitiv nou"
    ],
    "Curierat_Posta": [
        "colet blocat la vamă - taxe neplătite (Posta Romana)",
        "livrare eșuată - adresă incompletă (Fan Courier/Cargus/DPD)",
        "link urmărire colet (Sameday Easybox)",
        "retur bani pentru produs vândut pe OLX"
    ],
    "Utilitati_Stat": [
        "poprire cont ANAF / declarație unică",
        "factură scadentă (Enel/E.ON/Digi/Hidroelectrica)",
        "expirare rovinietă (CNAIR)",
        "alertă RO-ALERT falsă cu link",
        "rambursare contribuții sănătate (CNAS)"
    ],
    "Social_Scams": [
        "investiții crypto cu profit garantat (Deepfake)",
        "oferta angajare part-time 'Like-uri pe TikTok/YouTube'",
        "metoda 'Accidentul' (Mama/Tata am nevoie de bani)",
        "voucher cadou eMAG/Kaufland/Lidl aniversar",
        "invitație nuntă fișier infectat"
    ]
}

TACTICS = [
    "URGENTA MAXIMA (Acționează acum sau pierzi banii)",
    "AUTORITATE (Mesaj de la Poliție/ANAF/Director)",
    "CURIOZITATE (Vezi cine te-a spionat / Poze secrete)",
    "OPORTUNITATE (Voucher, Bani gratis, Investiție)",
    "CONFIRMARE (Confirmă o comandă pe care nu ai făcut-o)"
]


def clean_json_text(text):
    text = text.strip()
    # Eliminăm blocuri de cod markdown
    if "```" in text:
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match: text = match.group(1)
    return text.strip()


def get_prompt(batch_size, topic, tactic, use_diacritics):
    # Instrucțiune dinamică pentru diacritice
    if use_diacritics:
        style_instruction = "Use correct Romanian grammar WITH diacritics (ă, î, ș, ț, â)."
    else:
        style_instruction = "Write in Romanian but DO NOT use diacritics (use a, i, s, t instead). Simulates cheap SMS gateways."

    return f"""
    Role: Expert Cybersecurity Data Generator.
    Language: Romanian.
    Style: {style_instruction}
    Topic: {topic}
    Psychological Tactic: {tactic}

    Generate {batch_size} entries in a JSON Array. Mix EMAIL and SMS types.

    GUIDELINES:

    1. **PHISHING Messages (Label 1)**:
       - **URL Rule**: MUST contain a realistic looking but FAKE malicious URL.
         - BAD Examples: '[http://bt-neo-update.com](http://bt-neo-update.com)', '[http://anaf-plati-online.net](http://anaf-plati-online.net)', 'bit.ly/secure-login', 'www.posta-romana-taxe.org'.
       - **SMS Rule**: Keep it SHORT (under 160 chars). Use urgency.
       - **Email Rule**: Include a subject line.

    2. **LEGITIMATE Messages (Label 0)**:
       - **URL Rule**: If a link is needed, use ONLY OFFICIAL domains.
         - GOOD Examples: 'www.bancatransilvania.ro', 'anaf.ro', 'emag.ro', 'digi.ro'.
       - **Tone**: Professional, informational, calm. No panic.
       - **Hard Negative**: Create messages that look like security alerts but are actually safe (e.g., "You logged in successfully").

    OUTPUT FORMAT (JSON Array of Objects):
    [
      {{"text": "Subiect: Actualizare. Contul dvs a fost suspendat. Deblocare: [http://bt24-secure-app.com](http://bt24-secure-app.com)", "label": 1, "type": "email", "source": "{topic}"}},
      {{"text": "Salut, codul tau de autentificare este 123456. Nu il divulga nimanui.", "label": 0, "type": "sms", "source": "{topic}"}}
    ]
    """


def save_batch(data_batch, filepath):
    df = pd.DataFrame(data_batch)
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=header, index=False, quoting=1)


def sync_generated_copy():
    if os.path.exists(OUTPUT_FILE) and not os.path.exists(OUTPUT_FILE_GENERATED):
        os.makedirs(os.path.dirname(OUTPUT_FILE_GENERATED), exist_ok=True)
        try:
            import shutil
            shutil.copyfile(OUTPUT_FILE, OUTPUT_FILE_GENERATED)
        except Exception as exc:
            print(f"[WARN] Nu pot copia in {OUTPUT_FILE_GENERATED}: {exc}")


def has_valid_api_key() -> bool:
    return bool(API_KEY) and API_KEY.lower() != "api"


def init_model():
    if not has_valid_api_key():
        return None
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as exc:
        print(f"[WARN] Nu pot initializa Gemini: {exc}")
        return None


def build_local_message(is_phishing: bool, use_diacritics: bool) -> dict:
    banks = ["Banca Transilvania", "BCR", "ING", "Raiffeisen"]
    couriers = ["Sameday", "Fan Courier", "Cargus", "DPD"]
    utilities = ["ENEL", "E.ON", "Digi", "Hidroelectrica"]
    legit_domains = ["bancatransilvania.ro", "ing.ro", "anaf.ro", "emag.ro"]
    bad_domains = ["bt24-secure-app.com", "ro-anaf-plati.net", "posta-romana-verify.org"]

    typ = random.choice(["sms", "email"])
    if is_phishing:
        brand = random.choice(banks + couriers + utilities)
        bad_url = f"http://{random.choice(bad_domains)}"
        subject = f"Subiect: {brand} - actiune urgenta"
        body = f"Contul dvs necesita verificare imediata. Accesati {bad_url} pentru confirmare."
    else:
        brand = random.choice(banks + couriers + utilities)
        good_url = f"https://{random.choice(legit_domains)}"
        subject = f"Subiect: Informare {brand}"
        body = f"Va informam ca puteti verifica detalii in siguranta pe {good_url}."

    text = f"{subject} {body}" if typ == "email" else body
    if use_diacritics:
        text = (
            text.replace("dvs", "dvs.")
            .replace("informare", "informare")
            .replace("verificare", "verificare")
        )
    return {
        "text": text,
        "label": int(is_phishing),
        "type": typ,
        "source": "local_fallback",
        "lang": "ro",
    }


def local_fallback_generate(current_count: int):
    remaining = max(0, TARGET_COUNT - current_count)
    if remaining == 0:
        return
    print(f"[WARN] Fallback local activ. Generez {remaining} mostre.")
    batch = []
    for i in range(remaining):
        is_phish = random.random() < 0.5
        use_diacritics = random.random() > 0.4
        batch.append(build_local_message(is_phish, use_diacritics))
        if len(batch) >= BATCH_SIZE:
            save_batch(batch, OUTPUT_FILE)
            batch = []
    if batch:
        save_batch(batch, OUTPUT_FILE)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Verificăm unde am rămas
    current_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            current_count = len(df)
            print(f"[INFO] Se continuă de la {current_count} mesaje.")
        except:
            pass

    model = init_model()

    if model is None:
        local_fallback_generate(current_count)
        sync_generated_copy()
        print(f"Mesaje salvate in {OUTPUT_FILE}")
        return

    pbar = tqdm(total=TARGET_COUNT, initial=current_count)

    while current_count < TARGET_COUNT:
        # 1. Alegem Topic
        cat_key = random.choice(list(TOPICS_RO.keys()))
        topic = random.choice(TOPICS_RO[cat_key])

        # 2. Alegem Tactica
        tactic = random.choice(TACTICS)

        # 3. Alegem Diacritice (60% cu, 40% fără - pentru realism)
        use_diacritics = random.random() > 0.4

        try:
            prompt = get_prompt(BATCH_SIZE, topic, tactic, use_diacritics)
            response = model.generate_content(prompt)

            # Curățare și validare JSON
            json_str = clean_json_text(response.text)

            # Uneori AI-ul uită să închidă lista, încercăm să reparăm
            if not json_str.endswith("]"):
                json_str += "]"
            if not json_str.startswith("["):
                json_str = "[" + json_str

            batch_data = json.loads(json_str)

            valid_data = []
            for item in batch_data:
                # Validare chei
                if all(k in item for k in ("text", "label")):
                    # Adăugăm metadate utile pentru analiză
                    item["lang"] = "ro"
                    item["category"] = cat_key
                    item["has_diacritics"] = use_diacritics
                    if "type" not in item:
                        item["type"] = "mixed"
                    if "source" not in item:
                        item["source"] = "ai_generated"
                    valid_data.append(item)

            if valid_data:
                save_batch(valid_data, OUTPUT_FILE)
                count = len(valid_data)
                current_count += count
                pbar.update(count)

                # Feedback vizual mic
                d_status = "Diacritice" if use_diacritics else "Fara Diacritice"
                pbar.set_description(f"{d_status} | {topic[:15]}...")

            time.sleep(1.5)  # Evităm Rate Limit

        except json.JSONDecodeError:
            print(" [!] Eroare JSON (batch ignorat)")
        except Exception as e:
            print(f" [!] Eroare Generala: {e}. Pauza 5s...")
            time.sleep(5)

    pbar.close()
    sync_generated_copy()
    print(f"Mesaje salvate in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
