# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelu Fabian-Cătălin  
**Data:** 17.01.2026  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului "Detecție phishing în mesaje text". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, cu trasabilitate completă a transformărilor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```text
PROJECTPHISHING/
├── README.md
├── docs/
│   ├── datasets/
│   └── data_statistics.csv
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── preprocessing/
│   ├── data_acquisition/
│   └── neural_network/
├── config/
│   └── preprocessing_config.yaml
└── requirements.txt
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

- **Origine:** surse publice + date sintetice proprii.
- **Modul de achiziție:** ☐ Senzori reali / ☐ Simulare / ☑ Fișier extern / ☑ Generare programatică
- **Perioada / condițiile colectării:** decembrie 2025 – ianuarie 2026, procesare locală.
- **Fișiere sursă:**
  - `data/raw/emailreal.csv`
  - `data/raw/smsreal.csv`
  - `data/raw/phishing_ai_ro_only.csv` (copie sincronizată și în `data/generated/phishing_ai_ro_only.csv`)
  - `data/raw/phishing_ai_targeted_patch.csv`

### 2.2 Caracteristicile dataset-ului

- **Număr total de observații finale:** 40158
- **Număr de caracteristici:** 5
- **Tipuri de date:** ☐ Numerice / ☑ Categoriale / ☑ Text / ☐ Imagini
- **Format fișiere:** ☑ CSV / ☐ TXT / ☐ JSON / ☐ PNG

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| `text` | string | - | conținut mesaj email/SMS | lungime variabilă |
| `label` | int | - | eticheta de clasă | `{0=legit, 1=phishing}` |
| `type` | categorial | - | tip intrare | `{email, sms, mixed}` |
| `source` | categorial | - | sursa mesajului | surse publice + surse generate |
| `lang` | categorial | - | limba mesajului | `{en, ro, mixed}` |

**Fișier recomandat:** `docs/data_statistics.csv`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

- **Total:** 40158 observații
- **Distribuție clase:**
  - legit (`label=0`): 22624
  - phishing (`label=1`): 17534
- **Distribuție limbi:**
  - `en`: 22658
  - `ro`: 15000
  - `mixed`: 2500
- **Lungime text:**
  - medie: `1323.90`
  - mediană: `195`
  - percentila 95: `3518.15`

### 3.2 Analiza calității datelor

- Eliminare valori lipsă pe `text` și `label`.
- Curățare whitespace/newline/tab din mesaje.
- Eliminare duplicate pe `text`.
- Filtrare mesaje prea scurte (`min_text_len=6`).

### 3.3 Probleme identificate

- Variabilitate mare a lungimii mesajelor (SMS foarte scurte vs email-uri lungi).
- Mesaje foarte ambigue (context intern/BEC) care pot induce confuzii.
- Dezechilibru moderat de clasă (aprox. 56/44), acceptabil fără resampling în Etapa 3.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

- Eliminare duplicate: **1533** rânduri eliminate.
- Eliminare valori lipsă (`text`, `label`).
- Curățare text (`\n`, `\r`, `\t`, spații multiple).
- Filtru minim lungime text: `>5` caractere.

### 4.2 Transformarea caracteristicilor

- `lowercase: true` (conform `config/preprocessing_config.yaml`)
- `remove_urls: false` (URL-urile se păstrează pentru semnal phishing)
- Nu se aplică one-hot pe metadata în această etapă (metadata rămâne pentru analiză/documentare)
- Etichete convertite explicit la `int`

### 4.3 Structurarea seturilor de date

**Împărțire folosită:**
- `70%` train
- `15%` validation
- `15%` test

**Rezultate split:**
- `data/train/train.csv`: 28109
- `data/validation/validation.csv`: 6024
- `data/test/test.csv`: 6024

**Principii respectate:**
- Stratificare pe `label`
- `random_state=42`
- Fără leakage între split-uri

### 4.4 Salvarea rezultatelor preprocesării

- `data/raw/multilingualdataset.csv`
- `data/processed/processed.csv`
- `data/train/train.csv`
- `data/validation/validation.csv`
- `data/test/test.csv`
- configurare în `config/preprocessing_config.yaml`

---

##  5. Fișiere Generate în Această Etapă

- `src/data_acquisition/merge_all_datasets.py`
- `src/preprocessing/preprocess_and_split.py`
- `data/raw/multilingualdataset.csv`
- `data/processed/processed.csv`
- `data/train/train.csv`
- `data/validation/validation.csv`
- `data/test/test.csv`
- `docs/data_statistics.csv`

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + artefacte în `docs/`

---
