# Modul Data Acquisition

## Scop
Genereaza/combina date brute pentru detectia phishing.

## Scripturi
- `generate_ai_data.py`: genereaza mesaje AI RO (cu fallback local daca nu exista API key)
- `merge_all_datasets.py`: unifica email/SMS/AI in `data/raw/multilingualdataset.csv`

## Rulare
```bash
.venv/bin/python src/data_acquisition/generate_ai_data.py
.venv/bin/python src/data_acquisition/merge_all_datasets.py
```

## Output
- `data/raw/multilingualdataset.csv`
- `docs/data_statistics.csv`
- `data/generated/phishing_ai_ro_only.csv`

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`
