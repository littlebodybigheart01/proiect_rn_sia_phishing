# Modul Neural Network

## Model
DistilBERT multilingual pentru clasificare binara (legit/phishing).

## Fisiere cheie
- `model.py`: defineste arhitectura RN
- `train.py`: antrenare + evaluare + grafice
- `optimize.py`: optimizare hiperparametri (min 4 experimente)
- `save_untrained.py`: salveaza model neantrenat pentru Etapa 4

## Rulare
```bash
.venv/bin/python src/neural_network/save_untrained.py
.venv/bin/python src/neural_network/train.py
OPT_EPOCHS=10 OPT_MAX_EXPERIMENTS=4 .venv/bin/python src/neural_network/optimize.py
.venv/bin/python src/neural_network/report_assets.py
```

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`
