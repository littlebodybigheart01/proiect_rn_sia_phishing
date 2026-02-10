import os
import sys
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

sys.path.append(os.getcwd())

# --- CONFIGURARE ---
DEFAULT_MODEL_PATH = "models/phishing_distilbert_multilingual"
OPTIMIZED_MODEL_PATH = "models/phishing_distilbert_multilingual_optimized"


def resolve_model_path():
    if os.path.isdir(OPTIMIZED_MODEL_PATH):
        return OPTIMIZED_MODEL_PATH
    return DEFAULT_MODEL_PATH


MODEL_PATH = resolve_model_path()


def load_system():
    print("[INFO] Se încarcă modelul...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        print(f"[EROARE] Nu pot încărca modelul: {e}")
        return None, None


def predict_message(text, tokenizer, model):
    # 1. Preprocesare minimă
    text_clean = text.lower()

    # 2. Tokenizare
    encodings = tokenizer(
        [text_clean],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    # 3. Predicție
    output = model(encodings)
    logits = output.logits

    # Sigmoid pentru a obține probabilitatea (0.0 -> 1.0)
    prob = tf.nn.sigmoid(logits).numpy()[0][0]

    return prob


def main():
    tokenizer, model = load_system()
    if not model:
        return

    # --- DEFINIRE CULORI (ANSI Escape Codes) ---
    ROSU = "\033[91m"
    VERDE = "\033[92m"
    GALBEN = "\033[93m"
    ALBASTRU = "\033[94m"
    RESET = "\033[0m"  # Resetează culoarea la default
    # -------------------------------------------

    print("\n" + "=" * 60)
    print(f"{ALBASTRU}DETECTOR PHISHING (RO/EN) - TEST{RESET}")
    print("=" * 60)
    print("Scrie un mesaj pentru a-l verifica (scrie 'exit' sau apasa CTRL+C pentru a iesi).\n")

    while True:
        user_input = input(">> Mesaj: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        if not user_input.strip():
            continue

        score = predict_message(user_input, tokenizer, model)
        percent = score * 100

        print("-" * 40)

        # --- LOGICA DE CULORI ---
        if percent > 75:
            # Cazul de Phishing CLAR (Peste 75%) -> ROȘU
            print(f"{ROSU}[PHISHING] - Siguranță: {percent:.2f}%{RESET}")
            print(f"{ROSU}ATENȚIE! Acest mesaj este periculos!{RESET}")

        elif percent < 25:
            # Cazul Legitim CLAR (Sub 25% șanse de phishing) -> VERDE
            print(f"{VERDE}[LEGITIM]  - Probabilitate Phishing: {percent:.2f}%{RESET}")
            print(f"{VERDE}Mesajul pare sigur.{RESET}")

        else:
            # Zona de MIJLOC (Între 25% și 75%) -> GALBEN
            # Aici modelul nu este 100% sigur, e bine să avertizezi utilizatorul
            print(f"{GALBEN}[SUSPECT / INCERT] - Probabilitate Phishing: {percent:.2f}%{RESET}")
            print(f"{GALBEN}Atenție: Modelul nu este sigur. Verifică manual link-ul!{RESET}")

        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
