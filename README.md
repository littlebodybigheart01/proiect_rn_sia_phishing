# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelu Fabian-Cătălin  
**Link Repository GitHub:** `https://github.com/littlebodybigheart01/proiect_rn_sia_phishing`  
**Data predării:** 19.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din specificațiile proiectului.

**Obiectiv principal:** antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea pe setul de test și integrarea în aplicație.

**Pornire obligatorie din Etapa 4:**
- State Machine definit
- cele 3 module funcționale
- contribuție date originale >= 40%

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

- [x] **State Machine** definit în `docs/state_machine.svg`
- [x] **Contribuție >=40% date originale** (`17500 / 40158 = 43.58%`)
- [x] **Modul 1** funcțional (`src/data_acquisition/`)
- [x] **Modul 2** cu arhitectură definită și model neantrenat (`models/untrained_model.h5`)
- [x] **Modul 3** funcțional (`app.py`)
- [x] **Tabelul Nevoie -> Soluție -> Modul** completat în README Etapa 4

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

```bash
python3 src/data_acquisition/merge_all_datasets.py
python3 src/preprocessing/preprocess_and_split.py
```

**Verificări folosite în proiect:**
- config unitar: `config/preprocessing_config.yaml`
- split: `70/15/15`, stratificat, `random_state=42`
- fișiere rezultate:
  - `data/train/train.csv` (28109)
  - `data/validation/validation.csv` (6024)
  - `data/test/test.csv` (6024)

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Cerințe îndeplinite:
1. [x] Model antrenat pe setul final (cu >=40% date originale)
2. [x] Min. 10 epoci (rulate 16, limită max 50 cu early stopping)
3. [x] Split stratificat 70/15/15
4. [x] Tabel hiperparametri + justificări
5. [x] Metrici test set peste prag:
   - Accuracy: **0.9887**
   - F1 macro: **0.9885**
6. [x] Model salvat în `models/trained_model.h5`
7. [x] Integrare UI cu inferență reală (`docs/screenshots/inference_real.png`)

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | `3e-5` | valoare stabilă pentru fine-tuning DistilBERT |
| Batch size | `16` | echilibru între stabilitate gradient și memorie GPU |
| Number of epochs | `50` max, `16` rulate | plafon mare + early stopping după min. 10 epoci |
| Optimizer | `Adam` | convergență bună pe task NLP binar |
| Loss function | `BinaryCrossentropy(from_logits=True)` | modelul produce logit unic (`num_labels=1`) |
| Activation / output | sigmoid pe logit la inferență | probabilitate phishing în [0,1] |
| Max sequence length | `128` | compromis performanță / memorie / latență |

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Cerințe îndeplinite:
1. [x] EarlyStopping (`patience=5`, restore best weights)
2. [x] ReduceLROnPlateau (`factor=0.2`, `patience=1`)
3. [x] Date targetate noi (contribuție originală + patch edge-cases)
4. [x] Curbe loss/accuracy salvate: `docs/loss_curve.png`
5. [x] Analiză erori în context aplicativ (secțiunea dedicată)

**Indicatori obținuți:**
- Accuracy: **0.9887**
- F1 macro: **0.9885**

---

### Nivel 3 – Bonus (până la 100%)

| **Activitate** | **Livrabil** | **Status** |
|----------------|--------------|------------|
| Comparare 2+ arhitecturi | DistilBERT vs baseline text | [x] |
| Export TFLite + benchmark latență | `models/final_model.tflite`, `results/tflite_latency.json` | [x] |
| Confusion Matrix + analiză exemple greșite | `docs/confusion_matrix.png`, `docs/results/misclassified_examples.csv` | [x] |

**Rezultate bonus:**
- DistilBERT: accuracy `0.9887`, F1 `0.9885`
- Baseline Text Model: accuracy `0.9620`, F1 `0.9612`
- TFLite baseline infer-model: `avg_latency_ms=0.0054` (benchmark local)

---

## Verificare Consistență cu State Machine (Etapa 4)

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `INPUT_CAPTURE` | Input text/OCR în UI |
| `PREPROCESS` | tokenizare DistilBERT (`max_length=128`) |
| `RN_INFERENCE` | forward pass model antrenat |
| `CONFIDENCE_CHECK` | scor + praguri low/high |
| `DISPLAY` | verdict SAFE/SUSPECT/PHISH în UI |
| `LOG_FEEDBACK` | feedback persistent în `data/feedback/user_feedback.csv` |

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?

Confuziile dominante sunt **false negative** pentru phishing cu semnale slabe (mesaje fără URL explicit, ton intern legitim).

### 2. Ce caracteristici ale datelor cauzează erori?

- texte extrem de scurte/ambigue
- mesaje tip BEC (business email compromise)
- conținut aproape "legit" lexical, dar malițios contextual

### 3. Ce implicații are pentru aplicația industrială?

În context operațional, false negatives sunt mai costisitoare decât false positives. De aceea, UI-ul include zonă `SUSPECT` și praguri configurabile.

### 4. Ce măsuri corective propuneți?

1. creșterea numărului de exemple BEC/CEO-fraud
2. calibrarea pragurilor în funcție de risc (departament / rol)
3. activarea unei bucle de retraining periodic din feedback validat

---

## Structura Repository-ului la Finalul Etapei 5

```text
PROJECTPHISHING/
├── app.py
├── config/
│   └── preprocessing_config.yaml
├── data/
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── screenshots/
│       └── inference_real.png
├── models/
│   ├── untrained_model.h5
│   └── trained_model.h5
├── results/
│   ├── training_history.csv
│   └── test_metrics.json
└── src/
    ├── data_acquisition/
    ├── preprocessing/
    └── neural_network/
```

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pregătire date (dacă refaceți pipeline-ul de la zero)

```bash
python3 src/data_acquisition/merge_all_datasets.py
python3 src/preprocessing/preprocess_and_split.py
```

### 3. Antrenare model

```bash
python3 src/neural_network/train.py
```

Output principal:
- `models/trained_model.h5`
- `results/training_history.csv`
- `results/test_metrics.json`
- `docs/loss_curve.png`
- `docs/confusion_matrix.png`

### 4. Evaluare pe test set (opțional separat)

```bash
python3 src/neural_network/evaluate.py
```

### 5. Lansare UI cu model antrenat

```bash
streamlit run src/app/main.py
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [x] State Machine existent
- [x] Contribuție >=40% date originale
- [x] Cele 3 module funcționale

### Preprocesare și Date
- [x] Dataset combinat și preprocesat
- [x] Split 70/15/15 stratificat
- [x] Config unic (`config/preprocessing_config.yaml`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [x] Min. 10 epoci rulate
- [x] Hiperparametri documentați
- [x] Accuracy >= 65% și F1 >= 0.60
- [x] Model salvat în `models/trained_model.h5`
- [x] Istoric salvat în `results/training_history.csv`

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [x] UI folosește model antrenat
- [x] Inferență reală funcțională
- [x] Screenshot `docs/screenshots/inference_real.png`

### Documentație Nivel 2 (dacă aplicabil)
- [x] Early stopping
- [x] LR scheduler
- [x] Curbe loss/val_loss
- [x] Analiză erori

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [x] Comparare arhitecturi
- [x] Export TFLite + benchmark
- [x] Confusion matrix + misclassified examples

### Verificări Tehnice
- [x] Scripturi rulează fără erori critice
- [x] Artefactele sunt generate în folderele cerute

### Verificare State Machine (Etapa 4)
- [x] Stările de inferență și logging sunt implementate în aplicație

### Pre-Predare
- [x] README completat fără placeholdere
- [x] Fișierele de rezultate sunt prezente

---

## Livrabile Obligatorii (Nivel 1)

- `models/trained_model.h5`
- `results/training_history.csv`
- `results/test_metrics.json`
- `docs/loss_curve.png`
- `docs/confusion_matrix.png`
- `docs/screenshots/inference_real.png`
- `docs/etapa5_antrenare_model.md`

---

## Predare și Contact

- commit recomandat: `Etapa 5 completă - antrenare model RN`
- tag recomandat: `v0.5-training`
