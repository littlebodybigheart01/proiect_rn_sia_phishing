## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Chelu Fabian-Cătălin |
| **Grupa / Specializare** | 632AB / FIIR |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/littlebodybigheart01/proiect_rn_sia_phishing |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow, Transformers, Streamlit, FastAPI) |
| **Domeniul Industrial de Interes (DII)** | Securitate cibernetică pentru fluxuri de comunicare digitală |
| **Tip Rețea Neuronală** | Transformer (DistilBERT fine-tuning) + baseline text model |

**Notă repository:** proiectul a fost inițial pe repository-ul `proiect_rn_phishing`. Din cauza limitelor Git LFS (blocaj la clone/checkout pentru fișiere versionate prin LFS), livrabilul final a fost migrat în repository-ul nou `proiect_rn_sia_phishing`, cu aceeași structură logică și documentație actualizată.

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 98.87% | 98.87% | +0.00% | ✓ |
| F1-Score (Macro) | ≥0.65 | 0.9885 | 0.9885 | +0.0000 | ✓ |
| Latență Inferență | ≤1500 ms (target local UI) | 1471.51 ms (avg) | 1471.51 ms (avg) | +0.00 ms | ✓ |
| Contribuție Date Originale | ≥40% | 43.58% | 43.58% | - | ✓ |
| Nr. Experimente Optimizare | ≥4 | 4 | 4 | - | ✓ |

*Notă latență:* valoarea este măsurare locală în UI (runtime), nu metrică exportată în `results/final_metrics.json`.

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Am folosit asistenți AI ca unelte de suport punctual (explicații, clarificări, verificări de consistență și formulare documentație), nu pentru a genera integral aplicația sau rezultatele experimentale. Codul, integrarea modulelor, rulările, interpretarea metricilor și deciziile finale au fost realizate de mine.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință | Confirmare |
|-----|---------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights random, NU pre-antrenat) | DA |
| 2 | Minimum **40% din date sunt contribuție originală** | DA |
| 3 | Codul este propriu sau sursele externe sunt citate explicit | DA |
| 4 | AI a fost folosit doar ca tool, nu ca sursă integrală de cod/dataset | DA |
| 5 | Pot explica și justifica fiecare decizie importantă | DA |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Organizațiile primesc zilnic volume mari de email/SMS, iar verificarea manuală a mesajelor suspecte produce întârzieri și erori. Mesajele de phishing moderne au devenit mai greu de identificat (text natural, scenarii contextuale, limbi multiple), ceea ce crește riscul de compromitere a conturilor, pierderi financiare și incidente operaționale.

Soluția propusă este un SIA de detecție phishing care oferă clasificare rapidă în UI, evidențiază cazurile incerte (`SUSPECT`) și colectează feedback uman pentru îmbunătățiri ulterioare.

### 2.2 Beneficii Măsurabile Urmărite

1. Detectare automată cu Accuracy > 90% și F1 > 0.90 pe test set.
2. Timp mediu de decizie sub 2 secunde pe inferență locală.
3. Reducerea expunerii la false negatives prin praguri configurabile.
4. Trasabilitate completă prin logging + feedback.
5. Demonstrație end-to-end reproductibilă pe Linux.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Detectarea mesajelor phishing în timp util | Clasificare text/OCR cu verdict SAFE/SUSPECT/PHISH | `src/neural_network/` + `src/app/` | Accuracy/F1 pe test set |
| Reducerea riscului operațional din mesaje ambigue | Praguri duale low/high + stare `SUSPECT` | `app.py` / `src/app/main.py` | FN controlat prin threshold tuning |
| Audit și îmbunătățire iterativă | Logging inferență + feedback utilizator + extensie browser | `src/api/server.py` + `data/feedback/` + `src/app/chrome_extension/` | fișiere feedback și istoric sesiune |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt (public + generat) |
| **Sursa concretă** | Email/SMS public + date generate local RO + patch targetat |
| **Număr total observații finale (N)** | 40158 |
| **Număr features** | 5 (`text`, `label`, `type`, `source`, `lang`) |
| **Tipuri de date** | Text + metadata categorială |
| **Format fișiere** | CSV |
| **Perioada colectării/generării** | Noiembrie 2025 – Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 40158 |
| **Observații originale (M)** | 17500 |
| **Procent contribuție originală** | 43.58% |
| **Tip contribuție** | Date sintetice + patch-uri țintite |
| **Locație cod generare** | `src/data_acquisition/generate_ai_data.py` |
| **Locație date originale** | `data/generated/`, `data/raw/phishing_ai_targeted_patch.csv` |

**Descriere metodă generare/achiziție:**

Datele originale au fost generate programatic pentru scenarii de phishing frecvente în limba română (bancar, curierat, utilități, social engineering), apoi completate cu patch-uri targetate pentru cazuri dificile (typosquatting, CEO fraud, call-back scam). Datele au fost unificate cu surse publice, deduplicate și filtrate pentru consistență.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 28109 |
| Validation | 15% | 6024 |
| Test | 15% | 6024 |

**Preprocesări aplicate:**
- lowercasing
- curățare whitespace/newline/tab
- eliminare duplicate pe text
- eliminare intrări invalide (`text`, `label`)
- split stratificat cu `random_state=42`

**Referințe fișiere:** `config/preprocessing_config.yaml`, `docs/data_statistics.csv`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python, Pandas | Generare/unificare dataset, statistici dataset | `src/data_acquisition/` |
| **Neural Network** | TensorFlow + Transformers | Antrenare, evaluare, optimizare model phishing | `src/neural_network/` |
| **Web Service / UI** | Streamlit + FastAPI + Chrome Extension | Inferență interactivă, feedback, scan din browser | `app.py`, `src/app/`, `src/api/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.svg`

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare input text/OCR | pornire aplicație | input disponibil |
| `PREPROCESS` | Tokenizare/normalizare input | input primit | date gata de inferență |
| `INFERENCE` | Forward pass model RN | input preprocesat | scor calculat |
| `DECISION` | Aplicare praguri low/high | scor disponibil | bucket SAFE/SUSPECT/PHISH |
| `OUTPUT` | Afișare verdict în UI | bucket calculat | feedback/continuare |
| `LOG_FEEDBACK` | Persistență feedback și istoric | feedback trimis | revenire `IDLE` |
| `ERROR` | Tratare erori model/OCR/API | excepție detectată | reset/revenire `IDLE` |

**Justificare:** Structura separă clar inferența de decizie și de feedback, ceea ce permite control operațional (threshold tuning), auditabilitate și extindere facilă cu extensia browser/API.

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| Prag decizie | prag unic implicit | prag dual (`low/high`) | control mai bun FN/FP |
| Stare decizie incertă | absentă | `SUSPECT` | evită verdict rigid în zona ambiguă |
| Logging | feedback minim | feedback + latență + bucket | audit și analiză post-rulare |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Model principal: `TFDistilBertForSequenceClassification` (`distilbert-base-multilingual-cased`) cu `num_labels=1` pentru clasificare binară.

Flux:
1. tokenizare (`max_length=128`)
2. encoder DistilBERT
3. head clasificare (logit)
4. sigmoid în inferență pentru scor phishing

Model secundar (benchmark): `baseline_text_model` (TextVectorization + Embedding + Dense) antrenat de la zero.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | `5e-5` | best F1 dintre experimentele Etapa 6 |
| Batch Size | `16` | compromis bun între stabilitate și timp de train |
| Epochs | `10` (experimente opt.) | comparabilitate între experimente |
| Optimizer | Adam | standard robust pentru fine-tuning transformer |
| Loss Function | BinaryCrossentropy (`from_logits=True`) | setup corect pentru logit binar |
| Regularizare | callbacks (`EarlyStopping`, `ReduceLROnPlateau`) | reduce overfitting și stabilizează convergența |
| Early Stopping | activ (train principal) | oprește când `val_loss` nu mai îmbunătățește |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | LR=3e-5, BS=16 | 0.9885 | 0.9883 | 1697.94 s | referință |
| Exp 1 | LR=1e-5, BS=16 | 0.9872 | 0.9870 | 1696.41 s | ușor sub baseline |
| Exp 2 | LR=5e-5, BS=16 | **0.9887** | **0.9885** | 1698.14 s | best |
| Exp 3 | LR=3e-5, BS=4 | 0.9852 | 0.9850 | 2718.99 s | timp mai mare, scor mai slab |
| **FINAL** | `lr_5e-5` | **0.9887** | **0.9885** | ~1698 s | model ales pentru livrare |

**Justificare model final:** configurația `lr_5e-5` maximizează F1 macro fără cost suplimentar de timp față de baseline.

**Referințe fișiere:** `results/optimization_experiments.csv`, `results/final_metrics.json`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 0.9887 | ≥0.70 | ✓ |
| **F1-Score (Macro)** | 0.9885 | ≥0.65 | ✓ |
| **Precision (Macro)** | 0.9889 | informativ | ✓ |
| **Recall (Macro)** | 0.9881 | informativ | ✓ |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 0.9885 | 0.9887 | +0.0002 |
| F1-Score | 0.9883 | 0.9885 | +0.0002 |

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

| Aspect | Observație |
|--------|------------|
| **Clasa cu performanță mai bună** | Legit (Precision 0.9880, Recall 0.9920) |
| **Clasa sensibilă** | Phishing (Precision 0.9897, Recall 0.9844) |
| **Confuzii frecvente** | Mesaje phishing ambigue, fără indicatori clasici (BEC-like) |
| **Dezechilibru clase** | moderat (56.34% legit / 43.66% phishing), gestionat fără resampling |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație |
|---|--------------------------|--------------|-------------|-----------------|-----------|
| 1 | mesaj promo tip „junk mail” | legit | phishing | pattern lexical slab suspect | FN |
| 2 | mesaj intern urgent (BEC style) | legit | phishing | ton legitim, fără URL | FN critic |
| 3 | text foarte scurt/ambiguu | legit | phishing | informație insuficientă | FN |
| 4 | text zgomotos/grămatical alterat | legit | phishing | robusteză redusă la noisy text | FN |
| 5 | mesaj generic marketing | legit | phishing | semnale slabe de atac | FN |

### 6.4 Validare în Context Industrial

Rezultatele obținute depășesc pragurile minime impuse de proiect, confirmând utilitatea modelului pentru triere operațională rapidă. În utilizare reală, prioritatea principală rămâne reducerea cazurilor de tip *False Negative* (mesaje phishing clasificate eronat ca legitime); din acest motiv, aplicația păstrează praguri de decizie configurabile și categoria intermediară `SUSPECT`.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| Model încărcat | trained | standard/optimized selectabil | comparație rapidă |
| Decizie | simplificată | SAFE/SUSPECT/PHISH cu praguri | control risc |
| Stats UI | basic | sesiune, p50/p90 latență, bucket counts | observabilitate |
| Extensie browser | absentă | scan din click dreapta + popup | demo real |
| Feedback | minim | feedback persistent și exportabil | audit |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

**Ce se vede în screenshot și ce demonstrează:**

- input text real introdus în interfață + verdict final (`SAFE` / `SUSPECT` / `PHISH`)
- probabilitate de phishing (scor numeric) calculată de modelul optimizat
- panou de statistici sesiune (latență medie, p90 latență, distribuție verdicturi)
- mecanism de feedback (`corect` / `greșit`) care scrie în `data/feedback/user_feedback.csv`

Acest screenshot demonstrează că UI-ul folosește modelul optimizat în inferență reală, nu un model dummy.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/demo_end_to_end.mp4`

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat vizibil |
|-----|---------|------------------|
| 1 | Pornire aplicație (`streamlit run app.py`) | UI disponibil local, model încărcat |
| 2 | Introducere text / OCR pe imagine | text extras și trimis către inferență |
| 3 | Inferență model optimizat | verdict + scor afișate în interfață |
| 4 | Feedback utilizator | înregistrare în `data/feedback/user_feedback.csv` |
| 5 | Verificare stats sesiune | actualizare latență și distribuție predicții |

**Latență end-to-end observată în sesiunea documentată:** ~1471.51 ms (medie locală, hardware-ul de test al studentului).
**Data demonstrației:** 10.02.2026.

---

## 8. Structura Repository-ului Final

```text
PROJECTPHISHING/
├── README.md
├── Chelu_Fabian_Catalin_632ab_README_Proiect_RN.md
├── app.py
├── run_all.sh
├── run_services.sh
├── stop_services.sh
├── requirements.txt
├── config/
│   ├── preprocessing_config.yaml
│   ├── preprocessing_params.pkl
│   └── optimized_config.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── feedback/
├── docs/
│   ├── etapa3_analiza_date.md
│   ├── etapa4_arhitectura_SIA.md
│   ├── etapa5_antrenare_model.md
│   ├── etapa6_optimizare_concluzii.md
│   ├── state_machine.svg
│   ├── confusion_matrix_optimized.png
│   ├── screenshots/
│   ├── optimization/
│   ├── results/
│   └── demo/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   │   ├── combine_datasets.py
│   │   ├── data_cleaner.py
│   │   ├── feature_engineering.py
│   │   ├── data_splitter.py
│   │   └── preprocess_and_split.py
│   ├── neural_network/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── optimize.py
│   │   └── visualize.py
│   ├── api/
│   └── app/
│       ├── main.py
│       └── chrome_extension/
├── results/
├── models/   (modele mari, livrate prin Google Drive)
└── .gitignore
```

### Legendă Progresie pe Etape

| Etapă | Data folosită | Livrabil principal |
|------|---------------|--------------------|
| Etapa 3 | 21.11.2025 | `docs/etapa3_analiza_date.md` |
| Etapa 4 | 05.12.2025 | `docs/etapa4_arhitectura_SIA.md` |
| Etapa 5 | 19.12.2025 | `docs/etapa5_antrenare_model.md` |
| Etapa 6 | 16.01.2026 | `docs/etapa6_optimizare_concluzii.md` |

### Convenție Tag-uri Git

| Tag | Etapa | Status |
|-----|-------|--------|
| `v0.3-data-ready` | Etapa 3 | creat |
| `v0.4-architecture` | Etapa 4 | creat |
| `v0.5-model-trained` | Etapa 5 | creat |
| `v0.6-optimized-final` | Etapa 6 | creat |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

- Python >= 3.10
- Linux (mediu principal folosit în proiect)
- pip + venv

### 9.2 Instalare

```bash
git clone https://github.com/littlebodybigheart01/proiect_rn_sia_phishing
cd proiect_rn_sia_phishing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
bash run_all.sh
```

### 9.4 Rulare UI + API (Linux)

```bash
bash run_services.sh
# oprește cu:
bash stop_services.sh
```

**Notă:** scripturile `.sh` (`run_all.sh`, `run_services.sh`, `stop_services.sh`) au fost adăugate explicit deoarece proiectul a fost dezvoltat și testat pe Linux, pentru pornire/oprire rapidă și reproductibilitate.

### 9.5 Modele mari (Google Drive)

Fișierele model sunt mari și nu sunt incluse integral în GitHub.

- Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`
- După download, copiați conținutul în `models/`.

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv | Target | Realizat | Status |
|----------|--------|----------|--------|
| Accuracy pe test set | ≥0.70 | 0.9887 | ✓ |
| F1 macro pe test set | ≥0.65 | 0.9885 | ✓ |
| Contribuție date originale | ≥40% | 43.58% | ✓ |
| Experimente optimizare | ≥4 | 4 | ✓ |
| 3 module funcționale | obligatoriu | implementate | ✓ |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. Erori reziduale pe phishing ambiguu semantic (BEC-like).
2. Latență mai mare față de modele ultra-lightweight.
3. Dependință de calitatea textului extras OCR.

### 10.3 Lecții Învățate (Top 5)

1. Datele targetate cresc calitatea mai mult decât tuning-ul agresiv.
2. Pragurile duale sunt utile operațional.
3. Logging + feedback ajută pentru iterații controlate.
4. Baseline-ul from-scratch este util pentru validare comparativă.
5. Scripturile de rulare reduc erorile de prezentare în examen.

### 10.4 Retrospectivă

Dacă reluam proiectul, am introduce mai devreme un subset BEC dedicat și un modul de calibrare probabilistică pe validare pentru reducerea sistematică a false negatives.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|----------------------|-------------------|
| Short-term | calibrare praguri pe profile de risc | FN mai mic |
| Medium-term | retraining incremental din feedback validat | robustețe mai bună |
| Long-term | ensemble DistilBERT + model lexical rapid | latență/performanță mai bună |

---

## 11. Bibliografie

1. Devlin, J. et al., 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
2. Sanh, V. et al., 2019. DistilBERT, a distilled version of BERT. https://arxiv.org/abs/1910.01108
3. TensorFlow Documentation. https://www.tensorflow.org/
4. Hugging Face Transformers Documentation. https://huggingface.co/docs/transformers
5. Scikit-learn Metrics Documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
6. DistilBERT multilingual base model card. https://huggingface.co/distilbert/distilbert-base-multilingual-cased
7. DistilBERT model documentation (Transformers). https://huggingface.co/docs/transformers/model_doc/distilbert
8. A Beginner's Guide to Streamlit (GeeksforGeeks). https://www.geeksforgeeks.org/python/a-beginners-guide-to-streamlit/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] Accuracy ≥70% pe test set
- [x] F1-Score ≥0.65 pe test set
- [x] Contribuție ≥40% date originale
- [x] Minimum 4 experimente de optimizare
- [x] Confusion matrix generată și interpretată
- [x] State Machine definit și utilizat
- [x] Cele 3 module funcționale (Data Logging + RN + UI)
- [x] Demonstrație end-to-end finală în `docs/demo/demo_end_to_end.mp4`

### Repository și Documentație

- [x] README final completat (`Chelu_Fabian_Catalin_632ab_README_Proiect_RN.md`)
- [x] README-uri etape în `docs/`
- [x] Screenshots în `docs/screenshots/`
- [x] Structură repo conformă secțiunii 8
- [x] `requirements.txt` actualizat
- [x] Link GitHub actualizat
- [x] Mențiune Google Drive pentru modele mari

### Acces și Versionare

- [x] Repository accesibil
- [x] Tag-uri `v0.3`, `v0.4`, `v0.5` create
- [x] Tag `v0.6-optimized-final` creat

### Verificare Anti-Plagiat

- [x] Contribuție proprie date >=40%
- [x] Decizii tehnice documentate și argumentate
- [x] AI utilizat doar ca suport, nu ca sursă integrală

---

## Note Finale

**Versiune document:** FINAL pre-examen  
**Ultima actualizare:** 09.02.2026  
**Tag Git final:** `v0.6-optimized-final` (după push-ul final în repository)
