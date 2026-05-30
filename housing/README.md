# Toronto Shelter Demand Predictor

Predict shelter occupancy 1 day ahead and identify emerging homelessness risk by ward using Toronto Open Data, Environment Canada weather, and real-time bike share demand signals. Trained on NVIDIA DGX Spark (128GB unified memory, GB10 GPU) with XGBoost, LSTM, and Temporal Fusion Transformer models.

## Results

| Model | AUC-ROC | Accuracy | MAE | Within 5% | Training |
|-------|---------|----------|-----|-----------|----------|
| **XGBoost (GPU)** | **0.93** | **87%** | **2.41%** | **87.6%** | 0.7s |
| LSTM (GPU) | 0.91 | 86% | 3.52% | 84.2% | 45.4s |
| TFT (GPU) | — | — | — | — | Pending |

Test set: 16,412 samples (Jan-May 2026), trained on 108,923 samples (2021-2025).

## What It Does

**Predict** tomorrow's shelter occupancy rate and whether individual shelter programs will hit 100% capacity, using 52 features spanning weather, temporal patterns, lag indicators, and capacity history.

**Detect** early warning signals for emerging homelessness by ward — shifting from reactive "is this shelter full?" to proactive "where will new episodes emerge?" using 311 complaints, RentSafeTO scores, and central intake call volumes.

**Monitor** real-time demand pressure using Toronto Bike Share station availability as a proxy signal for neighbourhood activity patterns near shelter locations.

**Forecast** with multiple model architectures: gradient-boosted trees (fast, interpretable), LSTM sequences (captures temporal dependencies), and Temporal Fusion Transformer (attention-based multi-horizon).

## Project Structure

```
housing/
├── scripts/
│   ├── 01_download_data.sh            # Download shelter + weather data
│   ├── 02_explore_data.py             # Initial data exploration
│   ├── 03_prepare_features.py         # Feature engineering (52 features)
│   ├── 04_train_model.py              # XGBoost classifier + regressor (GPU)
│   ├── 05_train_lstm.py               # PyTorch LSTM sequence model (GPU)
│   ├── 06_dashboard.py                # Streamlit prediction dashboard
│   ├── 07_hourly_weather.py           # Hourly weather + time-of-day features
│   ├── 08_retrain_with_hourly.py      # Retrain XGBoost with hourly features
│   ├── 09_train_tft.py                # Temporal Fusion Transformer (GPU)
│   ├── 10_enrich_features.py          # RentSafeTO, fire inspections, cooling centres
│   ├── 11_early_warning.py            # Ward-level homelessness early warning
│   └── realtime_feeds.py              # Bike Share GBFS + live shelter locations
├── data/
│   ├── raw/                           # Raw CSVs (gitignored)
│   └── processed/                     # Train/test parquet files (gitignored)
├── models/                            # Trained model artifacts (gitignored)
├── docs/
│   └── datasets.md                    # Full Toronto Open Data catalogue (70+ datasets)
└── venv/                              # Python virtual environment
```

## Setup

```bash
cd housing
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib requests pyarrow xgboost torch streamlit
```

## Usage

### 1. Download data

Downloads shelter occupancy (2021-2026, ~80MB) from Toronto Open Data and daily weather from Environment Canada (Toronto Pearson station).

```bash
bash scripts/01_download_data.sh
```

### 2. Explore and prepare features

```bash
python3 scripts/02_explore_data.py
python3 scripts/03_prepare_features.py    # 52 features, train/test split at 2026-01-01
```

**Feature groups (52 total):**
- Time/calendar (9): day of week, month, cyclical encodings, weekend flag
- Categorical (2): sector, program model
- Current state (5): today's occupancy rate, spare beds, unavailable ratio
- Lag features (4): occupancy rate 1, 3, 7, 14 days ago
- Rolling statistics (8): 3/7/14/30-day mean and std of occupancy rate
- Trend features (3): 3-day and 7-day trend, day-over-day delta
- Capacity history (4): at-capacity today/yesterday, 7-day and 30-day at-capacity rate
- Program history (2): expanding mean occupancy and at-capacity rate
- Weather (15): temperature (max/min/mean), rain, snow, wind gust, heating degree days, cold snap indicators, rolling temperature trends

### 3. Train models

```bash
python3 scripts/04_train_model.py      # XGBoost on GPU (<1s)
python3 scripts/05_train_lstm.py       # PyTorch LSTM on GPU (45s)
python3 scripts/09_train_tft.py        # Temporal Fusion Transformer (GPU)
```

### 4. Enhance with additional data

```bash
python3 scripts/07_hourly_weather.py       # Hourly weather features
python3 scripts/08_retrain_with_hourly.py  # Retrain with hourly resolution
python3 scripts/10_enrich_features.py      # RentSafeTO, fire, cooling centres
```

### 5. Dashboard and early warning

```bash
streamlit run scripts/06_dashboard.py      # Occupancy prediction dashboard
python3 scripts/11_early_warning.py        # Ward-level risk assessment
```

### 6. Real-time feeds

```python
from scripts.realtime_feeds import get_bikeshare_availability, get_shelter_locations

# Bike Share station availability near shelters (updates every 30s)
stations = get_bikeshare_availability()

# Active shelter locations from CKAN
shelters = get_shelter_locations()
```

## Data Sources

### Primary
- **[Daily Shelter & Overnight Service Occupancy/Capacity](https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/)** — 2021-2026, ~180K rows, daily refresh
- **[Environment Canada Daily Weather](https://climate.weather.gc.ca/)** — Toronto Pearson (Station 51459), 2021-2026
- **[Environment Canada Hourly Weather](https://climate.weather.gc.ca/)** — Sub-daily temperature, wind, precipitation

### Enrichment
- **[RentSafeTO Apartment Building Evaluations](https://open.toronto.ca/dataset/apartment-building-evaluation/)** — 5,341 buildings, 50+ inspection categories
- **[Residential Fire Inspection Results](https://open.toronto.ca/dataset/residential-fire-inspection-results/)** — 124K records with violation codes
- **[Cooling/Warming Centres](https://open.toronto.ca/dataset/air-conditioned-and-cool-spaces-heat-relief-network/)** — Emergency weather shelters
- **[311 Service Requests](https://open.toronto.ca/dataset/311-service-requests-customer-initiated/)** — Property standards, heating, shelter-related complaints by ward
- **[Toronto Bike Share (GBFS)](https://tor.publicbikesystem.net/ube/gbfs/v1/)** — 1,035 stations, real-time availability every 30s
- **[Central Intake Calls](https://open.toronto.ca/dataset/central-intake-calls/)** — Daily demand pressure on shelter system

See [docs/datasets.md](docs/datasets.md) for the full catalogue of 70+ datasets across all refresh rates.

## Architecture

```
Toronto Open Data (CKAN API)     Environment Canada      Bike Share GBFS
        │                               │                       │
        ▼                               ▼                       ▼
  01_download_data.sh ──────────────────┘               realtime_feeds.py
        │                                                       │
        ▼                                                       │
  03_prepare_features.py ── 52 features ◄───── 10_enrich ◄──────┘
        │
        ├──► 04_train_model.py ──── XGBoost on GPU (0.7s)
        ├──► 05_train_lstm.py ───── PyTorch LSTM on GPU (45s)
        ├──► 09_train_tft.py ────── Temporal Fusion Transformer
        │
        ├──► 06_dashboard.py ────── Streamlit occupancy predictions
        └──► 11_early_warning.py ── Ward-level risk assessment
```

## Hardware

- **MacBook Pro**: data prep, base model training, dashboard
- **NVIDIA DGX Spark** (GB10 GPU, 128GB unified memory): GPU-accelerated XGBoost (`device="cuda"`), PyTorch LSTM/TFT training, real-time inference
