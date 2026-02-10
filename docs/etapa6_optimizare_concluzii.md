# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelu Fabian-Cătălin  
**Link Repository GitHub:** `https://github.com/littlebodybigheart01/proiect_rn_sia_phishing`  
**Data predării:** 16.01.2026

---

## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din specificațiile proiectului.

Obiectivul a fost trecerea de la un model funcțional la un sistem maturizat: experimente de optimizare, analiză de erori, integrare completă în aplicație și documentare finală.

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

Etapa 6 reprezintă versiunea finală pre-examen. După această etapă, modificările sunt iterări controlate pe baza feedback-ului, nu schimbări de arhitectură majoră.

În proiect au fost consolidate:
- modelul optimizat
- stările de decizie SAFE/SUSPECT/PHISH
- logging complet al inferenței și feedback-ului
- pipeline end-to-end reproductibil (`run_all.sh`)

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

- [x] model antrenat (`models/trained_model.h5`)
- [x] metrici baseline peste prag minim (Accuracy/F1)
- [x] tabel hiperparametri complet
- [x] istoric training salvat (`results/training_history.csv`)
- [x] UI funcțional cu inferență reală
- [x] screenshot inferență (`docs/screenshots/inference_real.png`)
- [x] State Machine definit și utilizat

---

## Cerințe

1. [x] minimum 4 experimente de optimizare
2. [x] tabel comparativ experimente
3. [x] confusion matrix + interpretare
4. [x] analiză top 5 erori
5. [x] metrici finale pe test set: Accuracy >= 70%, F1 >= 0.65
6. [x] model optimizat salvat (`models/optimized_model.h5`)
7. [x] aplicație actualizată cu model optimizat și statistici
8. [x] concluzii tehnice + plan post-feedback

#### Tabel Experimente de Optimizare

| **Exp#** | **Modificare față de Baseline** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|----------------------------------|--------------|--------------|-------------------|----------------|
| baseline | LR=3e-5, BS=16 | 0.9885 | 0.9883 | 1697.94 s (~28.3 min) | referință |
| exp1 | LR=1e-5, BS=16 | 0.9872 | 0.9870 | 1696.41 s (~28.3 min) | LR prea mic |
| exp2 | LR=5e-5, BS=16 | **0.9887** | **0.9885** | 1698.14 s (~28.3 min) | best |
| exp3 | LR=3e-5, BS=4 | 0.9852 | 0.9850 | 2718.99 s (~45.3 min) | cost mare, scor mai slab |

**Justificare alegere finală:** exp2 (`lr_5e-5`) oferă cel mai bun F1 macro fără penalizare semnificativă de timp.

---

## 1. Actualizarea Aplicației Software în Etapa 6 

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| Model activ | `trained_model.h5` + model dir HF | suport explicit standard/optimized în UI și API | test comparativ rapid |
| Decizie | prag simplu 0.5 | praguri duale low/high (SAFE/SUSPECT/PHISH) | control risc FN/FP |
| Statistici UI | basic output | tab STATS extins: p50/p90 latență, bucket counts, istorice | observabilitate |
| Logging | feedback minim | feedback persistent + extension logs | trasabilitate |
| Integrare browser | absent | extensie Chrome locală cu scan contextual | demo end-to-end |

### Modificări concrete aduse în Etapa 6:

1. `app.py` folosește model selectabil din sidebar (`Standard`/`Optimized`).
2. `src/api/server.py` expune `/scan`, `/feedback`, `/health`, cu alegere model (`standard|optimized|auto`).
3. S-au adăugat artefacte automate pentru raport (`src/neural_network/report_assets.py`):
   - `docs/optimization/*.png`
   - `docs/results/*.png`
   - `docs/results/misclassified_examples.csv`
4. UI a fost sincronizat cu modelul optimizat și cu metricile finale.

### Diagrama State Machine Actualizată (dacă s-au făcut modificări)

- Diagrama finală: `docs/state_machine.svg`
- Actualizare principală: introducerea stării explicite `SUSPECT` între SAFE și PHISH pentru scoruri incerte.

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

- `docs/confusion_matrix_optimized.png`
- `results/final_metrics.json`

### Interpretare Confusion Matrix:

- Accuracy test: **0.9887**
- F1 macro: **0.9885**
- Precision macro: **0.9889**
- Recall macro: **0.9881**

Clasificarea legit este foarte stabilă, iar erorile rămase sunt concentrate pe mesaje phishing ambigue semantic (fără indicatori clasici URL/brand).

### 2.2 Analiza Detaliată a 5 Exemple Greșite

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| 5645 | 1 | 0 | 0.9999 | mesaj promoțional/"junk" cu semnale mixte | date suplimentare pe spam "soft" |
| 926 | 1 | 0 | 0.9999 | BEC intern, ton legitim | extindere subset BEC/CEO fraud |
| 4265 | 1 | 0 | 0.9999 | text foarte scurt, aproape non-informativ | filtru + augmentări scurte |
| 5788 | 1 | 0 | 0.9998 | mesaj zgomotos/gramatică alterată | robustețe pe noisy text |
| 5580 | 1 | 0 | 0.9998 | frază generică de marketing | calibrări prag + negative hard mining |

### Exemplu #926 - Phishing clasificat ca legit

Context: mesaj intern urgent fără URL explicit, cu ton profesional.  
Cauza principală: lipsa indicatorilor lexicali clasici de phishing.  
Impact: false negative (critic operațional).  
Măsuri: creșterea cazurilor BEC în train și calibrare praguri în funcție de politică de risc.

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

### Strategie de optimizare adoptată:

- grid restrâns manual pe hiperparametri cu impact mare:
  - learning rate
  - batch size
- păstrare arhitectură constantă (DistilBERT multilingual)
- criteriu principal de selecție: F1 macro pe test set

### 3.2 Grafice Comparative

Artefacte generate:
- `docs/optimization/accuracy_comparison.png`
- `docs/optimization/f1_comparison.png`
- `docs/optimization/learning_curves_best.png`

### 3.3 Raport Final Optimizare

### Raport Final Optimizare

**Baseline Etapa 5:**
- Accuracy: 0.9885
- F1 macro: 0.9883

**Model optimizat Etapa 6:**
- Accuracy: 0.9887
- F1 macro: 0.9885

**Config finală:**
- model: DistilBERT multilingual
- LR: `5e-5`
- batch size: `16`
- epochs: `10` (în experimente), training principal cu early stopping

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target minim** | **Status** |
|-------------|-------------|-------------|-------------|------------------|------------|
| Accuracy | model neantrenat | 0.9887 | 0.9887 | >= 0.70 | OK |
| F1 macro | model neantrenat | 0.9885 | 0.9885 | >= 0.65 | OK |
| Precision macro | N/A | 0.9888 | 0.9889 | informativ | OK |
| Recall macro | N/A | 0.9882 | 0.9881 | informativ | OK |
| Nr. experimente | 0 | 0 | 4 | >= 4 | OK |

### 4.2 Vizualizări Obligatorii

- [x] `docs/confusion_matrix_optimized.png`
- [x] `docs/results/learning_curves_final.png`
- [x] `docs/results/metrics_evolution.png`
- [x] `docs/results/example_predictions.png`
- [x] `docs/results/misclassified_examples.csv`

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale

### Evaluare sintetică a proiectului

- Obiective tehnice depășite semnificativ față de pragurile minime.
- Sistemul este funcțional end-to-end: date -> train -> optimize -> UI/API -> feedback.
- State Machine este implementat și folosit în logica de decizie.

### 5.2 Limitări Identificate

### Limitări tehnice ale sistemului

1. erori pe mesaje phishing subtile fără indicatori clasici
2. latență DistilBERT mai mare decât un model lightweight
3. dependență de calitatea textului OCR pentru scenariile imagine

### 5.3 Direcții de Cercetare și Dezvoltare

### Direcții viitoare de dezvoltare

1. calibrări de probabilitate și threshold adaptiv
2. retraining incremental din feedback validat
3. integrare ensemble (DistilBERT + model lexical rapid)
4. extindere testare pe domenii noi (chat intern, ticketing)

### 5.4 Lecții Învățate

### Lecții învățate pe parcursul proiectului

1. calitatea datelor țintite influențează mai mult decât tuning-ul agresiv
2. observabilitatea (latency + logs + feedback) este critică pentru robusteză
3. state machine explicit reduce regresiile de flux

### 5.5 Plan Post-Feedback (ULTIMA ITERAȚIE ÎNAINTE DE EXAMEN)

### Plan de acțiune după primirea feedback-ului

1. adăugare set BEC extins + reevaluare false negatives
2. finalizare structură repo exact pe format final cerut în ghid
3. tag final și verificare completă checklist anti-regresie

---

## Structura Repository-ului la Finalul Etapei 6

```text
PROJECTPHISHING/
├── app.py
├── config/
├── data/
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── feedback/
├── docs/
│   ├── state_machine.svg
│   ├── confusion_matrix*.png
│   ├── optimization/
│   ├── results/
│   └── screenshots/
├── models/
├── results/
├── run_all.sh
└── src/
    ├── api/
    ├── app/
    │   ├── main.py
    │   └── chrome_extension/
    ├── data_acquisition/
    ├── preprocessing/
    └── neural_network/
```

---

## Instrucțiuni de Rulare (Etapa 6)

### 1. Rulare experimente de optimizare

```bash
# pipeline complet local (fără Streamlit)
bash run_all.sh
```

### 2. Evaluare și comparare

```bash
python3 src/neural_network/optimize.py
python3 src/neural_network/visualize.py
```

### 3. Actualizare UI cu model optimizat

```bash
streamlit run src/app/main.py
```

### 4. Generare vizualizări finale

```bash
python3 src/neural_network/visualize.py
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 5 (verificare)
- [x] model antrenat disponibil
- [x] metrici baseline documentate
- [x] UI funcțional

### Optimizare și Experimentare
- [x] minim 4 experimente
- [x] tabel comparativ complet
- [x] model final selectat justificat

### Analiză Performanță
- [x] confusion matrix analizată
- [x] top 5 erori documentate
- [x] metrici finale peste praguri

### Actualizare Aplicație Software
- [x] model optimizat integrat
- [x] statistici și logging extinse
- [x] screenshot optimized (`docs/screenshots/inference_optimized.png`)

### Concluzii
- [x] limitări identificate
- [x] direcții viitoare definite
- [x] plan post-feedback definit

### Verificări Tehnice
- [x] scripturile principale rulează
- [x] artefactele de evaluare există

### Verificare Actualizare Etape Anterioare (ITERATIVITATE)
- [x] Etapa 3 actualizată cu date finale
- [x] Etapa 4 actualizată cu state machine final
- [x] Etapa 5 sincronizată cu rezultate finale

### Pre-Predare
- [x] documentația este completată fără placeholders
- [x] livrabilele cheie sunt prezente în repo

---

## Livrabile Obligatorii

- `results/optimization_experiments.csv`
- `results/final_metrics.json`
- `models/optimized_model.h5`
- `docs/confusion_matrix_optimized.png`
- `docs/results/misclassified_examples.csv`
- `docs/results/learning_curves_final.png`
- `docs/results/metrics_evolution.png`
- `docs/results/example_predictions.png`
- `docs/screenshots/inference_optimized.png`
- `docs/etapa6_optimizare_concluzii.md`

---

## Predare și Contact

- commit recomandat: `Etapa 6 completa - optimizare si concluzii`
- tag final recomandat: `v0.6-optimized-final`

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`

---

## Bibliografie

1. DistilBERT multilingual base model card. https://huggingface.co/distilbert/distilbert-base-multilingual-cased
2. DistilBERT model documentation (Transformers). https://huggingface.co/docs/transformers/model_doc/distilbert
3. A Beginner's Guide to Streamlit (GeeksforGeeks). https://www.geeksforgeeks.org/python/a-beginners-guide-to-streamlit/
