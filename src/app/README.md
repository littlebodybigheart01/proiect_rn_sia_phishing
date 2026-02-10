# Modul UI (Streamlit)

## Scop
Interfata pentru inferenta phishing (text + OCR) si statistici.

## Rulare
```bash
.venv/bin/streamlit run src/app/main.py
```

## Input
- Text direct
- Imagine cu text (OCR)

## Output
- Scor phishing + status SAFE/PHISH/SUSPECT
- Grafice: loss curve, confusion matrix, optimizare

## Extensie Chrome
- Locație: `src/app/chrome_extension/`
- Încărcare: `chrome://extensions` -> `Load unpacked` -> selectează `src/app/chrome_extension/`

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`
