# Toronto Shelter Demand Predictor

Predict shelter occupancy 1-7 days ahead using Toronto Open Data, running inference on NVIDIA DGX Spark (128GB).

## Project Structure

```
housing/
├── scripts/           # Data pipeline and model scripts
│   ├── 01_download_data.sh   # Download shelter occupancy CSVs
│   └── 02_explore_data.py    # Initial data exploration
├── data/
│   ├── raw/           # Raw CSVs from Toronto Open Data (gitignored)
│   └── processed/     # Cleaned/feature-engineered data (gitignored)
├── models/            # Trained model artifacts (gitignored)
└── docs/
    └── datasets.md    # Full catalogue of Toronto Open Data datasets
```

## Setup (on DGX Spark)

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib requests

bash scripts/01_download_data.sh
python3 scripts/02_explore_data.py
```

## Data Sources

Primary: [Daily Shelter & Overnight Service Occupancy/Capacity](https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/) (2021-2025, ~70MB)

See [docs/datasets.md](docs/datasets.md) for the full dataset catalogue across all refresh rates.

## Target

- **Classification**: Will a shelter reach capacity tomorrow? (binary per shelter/program)
- **Regression**: Predicted occupancy rate by shelter for next 1-7 days
