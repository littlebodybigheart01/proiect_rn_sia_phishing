# Dataset Overview

Acest folder conține datele folosite în proiectul RN de detecție phishing.

## Structură

- `raw/`: surse brute (publice + patch-uri)
- `generated/`: contribuția originală generată local
- `processed/`: date curate după preprocesare
- `train/`, `validation/`, `test/`: split-uri finale 70/15/15
- `feedback/`: feedback colectat din UI/extensie

## Set final

- total observații: `40158`
- label `0` (legit): `22624`
- label `1` (phishing): `17534`
- contribuție originală: `17500` (`43.58%`)

## Pipeline

1. unificare: `python src/data_acquisition/merge_all_datasets.py`
2. preprocesare + split: `python src/preprocessing/preprocess_and_split.py`

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`
