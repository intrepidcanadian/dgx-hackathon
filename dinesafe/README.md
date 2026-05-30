# Toronto DineSafe Risk Predictor

Predict which restaurants will fail their next health inspection using Toronto Open Data, XGBoost, Nemotron NLP features, and a neighbourhood-level knowledge graph. Built for NVIDIA DGX Spark with GPU-accelerated training and txt2kg knowledge graph ingestion.

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Risk Classifier (pass/fail) | AUC-ROC | **0.82** |
| Risk Classifier | Avg Precision | 0.20 |
| Severity Regressor | MAE | **0.56** |
| Severity Regressor | R-squared | 0.18 |

58 features (31 base + 27 enrichment), trained on Toronto DineSafe inspection history.

## What It Does

**Predict** which food establishments are most likely to fail their next inspection based on violation history, establishment type, geographic clustering, and enrichment from 311 complaints, fire incidents, and apartment evaluations.

**Explain** why a neighbourhood has high risk using NLP features extracted from violation descriptions via Nemotron, clustered by violation type (pest, sanitation, temperature, structural, training, equipment, storage, waste).

**Visualize** risk at neighbourhood level with an intervention dashboard showing which areas need proactive health inspector visits.

**Connect** DineSafe data with apartment building evaluations and health hazard complaints in a knowledge graph (txt2kg) for cross-domain insight.

## Project Structure

```
dinesafe/
├── scripts/
│   ├── 01_download_data.sh            # Download DineSafe + enrichment data
│   ├── 02_explore_data.py             # Data exploration and distributions
│   ├── 03_prepare_features.py         # Feature engineering (31 base features)
│   ├── 04_train_model.py              # XGBoost classifier + regressor (GPU)
│   ├── 05_dashboard.py                # Streamlit risk dashboard
│   ├── 05_enrich_features.py          # Add 311 complaints, fire incidents
│   ├── 06_enrich_v2.py                # Expanded enrichment + severity prediction
│   ├── 07_nemotron_features.py        # NLP features via Nemotron + GPU clustering
│   ├── 08_improve_model.py            # Model improvements round 2
│   ├── 09_dashboard_neighbourhood.py  # Neighbourhood intervention dashboard
│   ├── 10_prepare_kg_documents.py     # Prepare docs for txt2kg ingestion
│   └── 11_ingest_kg.py                # Bulk-ingest into txt2kg knowledge graph
├── data/
│   ├── raw/                           # DineSafe CSV, 311, fire data
│   ├── processed/                     # Train/test parquet, feature columns
│   └── kg_documents/                  # Neighbourhood markdown for txt2kg
│       ├── bulk_documents.json
│       └── markdown/                  # Per-neighbourhood knowledge docs
└── models/
    ├── xgb_risk_final.json
    └── model_final_metadata.json
```

## Setup

```bash
pip install pandas numpy scikit-learn xgboost pyarrow requests

# On DGX Spark (for Nemotron NLP + knowledge graph)
ollama pull nemotron                  # NLP feature extraction
docker compose up -d                  # txt2kg for knowledge graph
```

## Usage

### 1. Download and prepare data

```bash
bash scripts/01_download_data.sh
python3 scripts/02_explore_data.py
python3 scripts/03_prepare_features.py
```

### 2. Enrich features

```bash
python3 scripts/05_enrich_features.py    # 311 complaints + fire incidents
python3 scripts/06_enrich_v2.py          # Expanded enrichment + geo clustering
python3 scripts/07_nemotron_features.py  # NLP violation categories (needs Ollama)
```

### 3. Train model

```bash
python3 scripts/04_train_model.py        # Base model
python3 scripts/08_improve_model.py      # Enhanced model with all enrichments
```

### 4. Dashboards

```bash
streamlit run scripts/05_dashboard.py                  # Establishment risk view
streamlit run scripts/09_dashboard_neighbourhood.py    # Neighbourhood intervention view
```

### 5. Knowledge graph (requires DGX Spark + txt2kg)

```bash
python3 scripts/10_prepare_kg_documents.py    # Generate neighbourhood docs
python3 scripts/11_ingest_kg.py               # Ingest into txt2kg
```

## Data Sources

### Primary
- **[DineSafe](https://open.toronto.ca/dataset/dinesafe/)** — Restaurant inspection results, violation details, severity ratings. Updated daily.

### Enrichment
- **[311 Service Requests](https://open.toronto.ca/dataset/311-service-requests-customer-initiated/)** — Pest, sanitation, and property standards complaints near food establishments
- **[Fire Inspection Results](https://open.toronto.ca/dataset/residential-fire-inspection-results/)** — Nearby fire code violations as building quality proxy
- **[Apartment Building Evaluations](https://open.toronto.ca/dataset/apartment-building-evaluation/)** — RentSafeTO scores for neighbourhood building quality
- **[Health Hazard Complaints](https://open.toronto.ca/dataset/health-hazard-complaint/)** — Pest, odour, and sanitation complaints

### Knowledge Graph Sources
Neighbourhood-level documents combining DineSafe patterns with apartment evaluations and health hazard data, ingested into txt2kg for cross-domain querying.

## Feature Engineering (58 features)

- **Establishment history** (12): fail rate, violation counts by severity, days since last inspection, inspection frequency, consecutive pass/fail streaks
- **Violation categories** (8): per-establishment rates for pest, sanitation, temperature, structural, training, equipment, storage, waste (via Nemotron NLP)
- **Geographic** (6): lat/lon grid cluster, cluster-level fail rate, nearby establishment count, neighbourhood risk score
- **Temporal** (5): month, quarter, day of week, days since last fail, cyclical encodings
- **Enrichment** (27): 311 complaint density, fire violation proximity, apartment building scores, health hazard complaint rate

## Hardware

- **MacBook Pro**: data prep, base model training, dashboards
- **DGX Spark (GB10 Blackwell)**: Nemotron NLP feature extraction via Ollama, GPU XGBoost training, txt2kg knowledge graph with local LLM triple extraction
