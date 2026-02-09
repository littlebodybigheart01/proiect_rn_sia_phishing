import os
import sys

# --- FIX CRITIC PENTRU COMPATIBILITATE ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

sys.path.append(os.getcwd())
from src.neural_network.model import build_model


UNTRAINED_MODEL_H5 = "models/untrained_model.h5"


def main():
    os.makedirs("models", exist_ok=True)
    model = build_model()
    # DistilBERT este subclassed -> salvam weights in H5 pentru cerinta Etapa 4
    model.save_weights(UNTRAINED_MODEL_H5)
    print(f"[OK] Untrained model saved: {UNTRAINED_MODEL_H5}")


if __name__ == "__main__":
    main()
