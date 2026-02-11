# It's Ham or Spam? (RO + EN)

## RO
Aplicatie locala de detectie phishing/spam pentru text si OCR, construita cu DistilBERT multilingual, UI Streamlit, API FastAPI si extensie Chrome.

### Functionalitati
- Detectie phishing pe text introdus manual.
- Detectie phishing pe text extras din imagini (OCR).
- Selectie model `standard` / `optimized` / `auto`.
- Logging de feedback utilizator.
- Integrare extensie Chrome pentru scanare din click dreapta.

### Metrici principale
| Model | Accuracy | F1 Macro | Precision Macro | Recall Macro |
|---|---:|---:|---:|---:|
| Standard (`results/test_metrics.json`) | 0.9887 | 0.9885 | 0.9888 | 0.9882 |
| Optimizat (`results/final_metrics.json`) | 0.9887 | 0.9885 | 0.9889 | 0.9881 |

### Modele in cloud
Modelele mari sunt stocate in Google Drive:
- https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing

Dupa download, plaseaza folderele in `models/`.

### Rulare
1. Pipeline complet:
```bash
bash run_all.sh
```
2. Pornire API + Streamlit:
```bash
bash run_services.sh
```
3. Oprire servicii:
```bash
bash stop_services.sh
```

### Extensia Chrome
1. Deschide `chrome://extensions`.
2. Activeaza Developer mode.
3. `Load unpacked` -> selecteaza `src/app/chrome_extension`.
4. Selecteaza text intr-o pagina -> click dreapta -> `Scan for spam`.

## EN
Local phishing/spam detector for text and OCR, built with DistilBERT multilingual, Streamlit UI, FastAPI backend, and a Chrome extension.

### Features
- Phishing detection for typed text.
- Phishing detection for OCR-extracted text from images.
- Model selection: `standard` / `optimized` / `auto`.
- User feedback logging.
- Chrome extension with right-click scan action.

### Main metrics
| Model | Accuracy | F1 Macro | Precision Macro | Recall Macro |
|---|---:|---:|---:|---:|
| Standard (`results/test_metrics.json`) | 0.9887 | 0.9885 | 0.9888 | 0.9882 |
| Optimized (`results/final_metrics.json`) | 0.9887 | 0.9885 | 0.9889 | 0.9881 |

### Model storage (cloud)
Large model artifacts are stored on Google Drive:
- https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing

After download, place model folders under `models/`.

### Run
1. Full pipeline:
```bash
bash run_all.sh
```
2. Start API + Streamlit:
```bash
bash run_services.sh
```
3. Stop services:
```bash
bash stop_services.sh
```

### Chrome extension
1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click `Load unpacked` and select `src/app/chrome_extension`.
4. Select text in a webpage -> right click -> `Scan for spam`.

## Paths
- App: `app.py`
- API: `src/api/server.py`
- Extension: `src/app/chrome_extension/`
- Stage docs: `docs/etapa3_analiza_date.md`, `docs/etapa4_arhitectura_SIA.md`, `docs/etapa5_antrenare_model.md`, `docs/etapa6_optimizare_concluzii.md`
- Final academic README: `Chelu_Fabian_Catalin_632ab_README_Proiect_RN.md`
