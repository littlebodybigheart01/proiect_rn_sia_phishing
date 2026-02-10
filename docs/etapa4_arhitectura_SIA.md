# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Chelu Fabian-Cătălin  
**Link Repository GitHub**  
`https://github.com/littlebodybigheart01/proiect_rn_sia_phishing`  
**Data:** 05.12.2025  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape din specificațiile proiectului.

În această etapă a fost livrat scheletul complet al SIA: pipeline end-to-end, model definit/compilat, UI funcțional și flux de stări documentat.

### IMPORTANT - Ce înseamnă "schelet funcțional":

**CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori.
- Pipeline-ul complet rulează end-to-end (date -> output UI/API).
- Modelul RN este definit și compilat.
- UI/Web layer primește input și returnează output.

**CE NU E NECESAR ÎN ETAPA 4:**
- Model cu performanță finală optimizată.
- Tuning complet de hiperparametri.
- Deployment cloud/producție.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea rapidă a mesajelor phishing în flux email/SMS | Clasificare binară cu scor probabilistic și verdict în UI în < 2 secunde | RN + UI |
| Reducerea riscului de click pe link-uri malițioase | Separare SAFE/SUSPECT/PHISH prin praguri configurabile | RN + UI + State Machine |
| Trasabilitate și învățare continuă din corecții utilizator | Logging feedback (`correct/wrong`) pentru audit și iterații de date | UI/API + Data Logging |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

#### Cum se calculează 40%:

- Total observații finale: **40158**
- Observații publice (email + sms): **22658**
- Observații originale (generate + patch targetat): **17500**
- Procent contribuție originală: **43.58%** (`17500 / 40158`)

Condiția minimă de 40% este îndeplinită.

#### Tipuri de contribuții acceptate (exemple din inginerie):

| **Tip contribuție** | **Aplicare în proiect** | **Dovada minimă** |
|---------------------|-------------------------|-------------------|
| Date sintetice prin metode avansate | Generare set RO pe scenarii phishing reale (bănci, curierat, utilități, social engineering) | `src/data_acquisition/generate_ai_data.py`, `data/raw/phishing_ai_ro_only.csv` (sincronizat în `data/generated/phishing_ai_ro_only.csv`) |
| Patch targetat pe cazuri dificile | Typosquatting, CEO fraud, callback scam, legit urgent work | `data/raw/phishing_ai_targeted_patch.csv` |

#### Declarație obligatorie în README:

### Contribuția originală la setul de date:

**Total observații finale:** 40158  
**Observații originale:** 17500 (43.58%)

**Tipul contribuției:**
- [ ] Date generate prin simulare fizică
- [ ] Date achiziționate cu senzori proprii
- [ ] Etichetare/adnotare manuală
- [x] Date sintetice prin metode avansate

**Descriere detaliată:**
Datele originale au fost generate controlat în limba română, pe template-uri de phishing observate în practică (bancar, curierat, utilități, investiții false, autoritate falsă) și completate cu patch-uri țintite pentru cazuri care în mod obișnuit produc erori de clasificare. Setul a fost apoi unificat cu surse publice (email + SMS), deduplicat și filtrat pentru calitate.

**Locația codului:** `src/data_acquisition/generate_ai_data.py`, `src/data_acquisition/merge_all_datasets.py`  
**Locația datelor:** `data/raw/phishing_ai_ro_only.csv` (sincronizat și în `data/generated/phishing_ai_ro_only.csv`), `data/raw/phishing_ai_targeted_patch.csv`

**Dovezi:**
- `docs/generated_vs_real.png`
- `docs/data_statistics.csv`

#### Exemple pentru "contribuție originală":
- [x] Date sintetice cu scenarii variate și control al etichetelor
- [x] Patch țintit pentru edge-cases relevante aplicației

#### Atenție - Ce NU este considerat "contribuție originală":

- simplă filtrare/normalizare pe date publice
- subset nerelevant extras dintr-un dataset public
- duplicare de date fără variație semantică

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Locație:** `docs/state_machine.svg`

Flux implementat:

```text
IDLE -> INPUT_CAPTURE -> PREPROCESS -> RN_INFERENCE -> CONFIDENCE_CHECK
      -> {SAFE | SUSPECT | PHISH} -> DISPLAY -> LOG_FEEDBACK -> IDLE
                     \-> ERROR -> IDLE
```

### Justificarea State Machine-ului ales:

Am ales acest State Machine deoarece aplicația este orientată pe clasificare text/OCR cu răspuns rapid și trasabilitate. Stările separă explicit pașii tehnici (captură, preprocesare, inferență, decizie) de pașii operaționali (afișare, logging, feedback), ceea ce face fluxul robust și ușor de extins.

Starea `SUSPECT` este utilă pentru zona de incertitudine, unde sistemul nu forțează verdict binar. Starea `ERROR` acoperă lipsa modelului, probleme OCR sau input invalid, apoi readuce aplicația în `IDLE` fără blocare.

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | N/A | Produce CSV cu datele voastre (inclusiv cele 40% originale). Codul rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py`, `src/neural_network/save_untrained.py` | N/A | Modelul RN este definit, compilat și poate fi încărcat. |
| **3. Web Service / UI** | `app.py`, `src/app/main.py`, `src/api/server.py` | N/A | Primește input de la user și afișează output-ul de clasificare. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [x] Scripturile rulează fără erori
- [x] Generează CSV compatibil cu preprocesarea
- [x] Asigură contribuție originală >= 40%
- [x] Au documentație minimă în cod și README-uri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [x] Arhitectură RN definită și compilată
- [x] Model salvat/reîncărcat (`models/untrained_model.h5`)
- [x] Justificare arhitectură în cod (`src/neural_network/model.py`)
- [x] Model neantrenat disponibil pentru etapa de schelet

#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [x] Interfață cu input text + OCR
- [x] Output verdict + scor
- [x] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`

---

## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

```text
PROJECTPHISHING/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   └── app/
├── docs/
│   ├── state_machine.svg
│   ├── generated_vs_real.png
│   └── screenshots/
├── models/
│   └── untrained_model.h5
├── config/
├── README.md
└── requirements.txt
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie -> Soluție -> Modul completat
- [x] Declarație contribuție >=40% completată
- [x] Dovezi contribuție originală prezente în `docs/`
- [x] Diagrama State Machine salvată în `docs/state_machine.svg`
- [x] Legendă/justificare State Machine inclusă

### Modul 1: Data Logging / Acquisition
- [x] Cod funcțional pentru generare/unificare date
- [x] CSV-uri valide generate
- [x] Contribuție originală verificabilă >=40%

### Modul 2: Neural Network
- [x] Arhitectură RN definită/compilată
- [x] Model neantrenat salvat (`models/untrained_model.h5`)

### Modul 3: Web Service / UI
- [x] UI pornește și permite inferență
- [x] Screenshot în `docs/screenshots/ui_demo.png`

---

**Predarea etapei (recomandat):**
- mesaj commit: `Etapa 4 completa - Arhitectura SIA functionala`
- tag: `v0.4-architecture`

---

## Modele (Google Drive)

Link modele: `https://drive.google.com/drive/folders/1Iv9m0HvrbKgabRXuzhahvfOC6t3-rpMu?usp=sharing`

---

## Bibliografie

1. DistilBERT multilingual base model card. https://huggingface.co/distilbert/distilbert-base-multilingual-cased
2. DistilBERT model documentation (Transformers). https://huggingface.co/docs/transformers/model_doc/distilbert
3. A Beginner's Guide to Streamlit (GeeksforGeeks). https://www.geeksforgeeks.org/python/a-beginners-guide-to-streamlit/
