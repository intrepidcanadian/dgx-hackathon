# Toronto Shelter Demand Predictor

Predict shelter occupancy 1 day ahead using Toronto Open Data and Environment Canada weather data, trained and served on an NVIDIA DGX Spark (128GB unified memory, GB10 GPU).

## Results

| Metric | XGBoost (GPU) | LSTM (GPU) |
|--------|--------------|------------|
| ROC AUC | **0.93** | 0.91 |
| Classification Accuracy | **87%** | 86% |
| Regression MAE | **2.41%** | 3.52% |
| Predictions within 5% | **87.6%** | 84.2% |
| Training time | 0.7s | 45.4s |

Test set: 16,412 samples (Jan-May 2026), trained on 108,923 samples (2021-2025).

## Project Structure

```
housing/
├── scripts/
│   ├── 01_download_data.sh       # Download shelter + weather data
│   ├── 02_explore_data.py        # Initial data exploration
│   ├── 03_prepare_features.py    # Feature engineering (52 features)
│   ├── 04_train_model.py         # XGBoost classifier + regressor (GPU)
│   └── 05_train_lstm.py          # PyTorch LSTM sequence model (GPU)
├── data/
│   ├── raw/                      # Raw CSVs (gitignored)
│   └── processed/                # Train/test parquet files (gitignored)
├── models/                       # Trained model artifacts (gitignored)
└── docs/
    └── datasets.md               # Full Toronto Open Data catalogue
```

## Setup (on DGX Spark)

```bash
git clone https://github.com/intrepidcanadian/dgx-hackathon.git housing
cd housing

python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib requests pyarrow xgboost torch
```

## Usage

### 1. Download data

Downloads shelter occupancy (2021-2026, ~80MB) from Toronto Open Data and daily weather from Environment Canada (Toronto Pearson station).

```bash
bash scripts/01_download_data.sh
```

### 2. Explore data

```bash
python3 scripts/02_explore_data.py
```

### 3. Prepare features

Builds 52 features from shelter occupancy and weather data. Predicts **tomorrow's** occupancy using only data available **today** (no data leakage). Outputs train/test parquet files split at 2026-01-01.

```bash
python3 scripts/03_prepare_features.py
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

### 4. Train XGBoost model (GPU)

Uses `device="cuda"` for GPU-accelerated gradient boosting. Trains both a binary classifier (will shelter hit 100%?) and a regressor (predicted occupancy rate). Early stopping enabled.

```bash
python3 scripts/04_train_model.py
```

### 5. Train LSTM model (GPU)

PyTorch 2-layer LSTM with dual prediction heads. Uses 14-day sequences per shelter program. Trains on GPU with batch size 512, cosine annealing LR schedule, and gradient clipping.

```bash
python3 scripts/05_train_lstm.py
```

## Data Sources

### Primary
- [Daily Shelter & Overnight Service Occupancy/Capacity](https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/) — 2021-2026, ~180K rows after filtering, daily refresh
- [Environment Canada Daily Weather](https://climate.weather.gc.ca/) — Toronto Pearson (Station 51459), 2021-2026

### Enrichment (planned)
- [RentSafeTO Apartment Building Evaluations](https://open.toronto.ca/dataset/apartment-building-evaluation/) — 5,341 buildings, 50+ inspection categories
- [Residential Fire Inspection Results](https://open.toronto.ca/dataset/residential-fire-inspection-results/) — 124K records with violation codes
- [Cooling/Warming Centres](https://open.toronto.ca/dataset/air-conditioned-and-cool-spaces-heat-relief-network/) — emergency weather shelters

See [docs/datasets.md](docs/datasets.md) for the full catalogue of 70+ datasets across all refresh rates (real-time, daily, weekly, monthly, semi-annual, annual).

## Architecture

```
Toronto Open Data (CKAN API)     Environment Canada
        │                               │
        ▼                               ▼
  01_download_data.sh ──────────────────┘
        │
        ▼
  02_explore_data.py ── initial analysis
        │
        ▼
  03_prepare_features.py ── 52 features, train/test split
        │
        ├──► 04_train_model.py ── XGBoost on GPU (0.7s)
        │
        └──► 05_train_lstm.py ── PyTorch LSTM on GPU
```

## Hardware

- NVIDIA DGX Spark — GB10 GPU, 128GB unified memory, aarch64
- XGBoost trains in <1s, LSTM uses GPU for batched sequence training
