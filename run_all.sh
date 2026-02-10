#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
pip install -r requirements.txt

python src/data_acquisition/merge_all_datasets.py
python src/preprocessing/preprocess_and_split.py
python src/neural_network/save_untrained.py
python src/neural_network/train.py

# Control optimizare: OPT_EPOCHS=10 OPT_MAX_EXPERIMENTS=4 ./run_all.sh
OPT_EPOCHS="${OPT_EPOCHS:-10}" OPT_MAX_EXPERIMENTS="${OPT_MAX_EXPERIMENTS:-4}" \
  python src/neural_network/optimize.py

python src/neural_network/report_assets.py

echo "[OK] Pipeline complet rulat (fara Streamlit)."
