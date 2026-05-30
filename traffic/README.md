# Toronto Traffic Intelligence Platform

Real-time traffic congestion prediction, event simulation, and commute optimization for Toronto using 346K turning movement counts, 336 live traffic cameras, live event data, and 6 enrichment datasets from Toronto Open Data. Combines XGBoost prediction (0.99 AUC) with Ollama vision-language models for live camera analysis on NVIDIA DGX Spark.

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Binary Congestion Classifier | AUC-ROC | **0.9937** |
| 4-Class Congestion Level | Accuracy | **87%** |
| Volume Regression | MAE | **4.6** (mean 419) |

Training: 276,923 samples, 30 features. Test: 69,231 samples.

## What It Does

**Predict** congestion at any Toronto intersection for any time/day using historical turning movement patterns.

**See** live traffic conditions by feeding 336 city camera images through a vision-language model (gemma3 on Ollama) that classifies congestion 0-3 with vehicle counts and incident detection.

**Simulate** the impact of events (concerts, sports games, road closures, construction) on an interactive map with before/after congestion visualization.

**Route** between any two points in Toronto with scored alternatives, transit comparison, and live TTC delay awareness.

**Optimize** your daily commute with 15-minute departure window analysis — know exactly when to leave to minimize drive time or catch the sweet spot after rush hour.

**Monitor** 24/7 via Hermes Agent with Telegram alerts when congestion changes, actionable route advice, and morning commute recommendations.

## Project Structure

```
traffic/
├── scripts/
│   ├── 01_prepare_traffic_data.py        # Pull CKAN data, engineer 30 features
│   ├── 01_prepare_traffic_data_gpu.py    # GPU-accelerated version (RAPIDS cuDF)
│   ├── 02_train_model.py                 # XGBoost: binary + 4-class + regression
│   ├── 02_train_model_gpu.py             # GPU training (device="cuda") + cuML
│   ├── 03_vlm_camera_analysis.py         # Feed cameras to Ollama VLM
│   ├── 04_dashboard.py                   # Streamlit dashboard
│   ├── 05_camera_pattern_analysis.py     # Full 336-camera pattern report
│   ├── 06_discover_traffic_patterns.py   # Time-series pattern discovery
│   ├── 07_hermes_traffic_monitor.py      # Scheduled monitoring + Telegram
│   ├── 08_enrich_traffic_data.py         # Pull 6 additional datasets
│   ├── 09_hermes_actionable_monitor.py   # Route advice + timing predictions
│   ├── 10_simulate_events.py             # Event injection + folium maps
│   ├── 11_route_advisor.py               # Point-to-point routing
│   ├── 12_commute_optimizer.py           # Daily commute optimization
│   ├── 13_enrich_events.py              # Real event data enrichment
│   ├── 14_vlm_feedback_loop.py          # VLM→model feedback + nowcasting
│   ├── 15_vlm_orchestrator.py           # Continuous VLM sweep + nowcast loop
│   └── 16_gnn_forecast.py               # Spatio-temporal GNN multi-horizon forecast
├── data/
│   ├── raw/                # Traffic cameras, turning movement counts
│   ├── processed/          # Train/test parquet, feature columns
│   ├── enrichment/         # TTC delays, collisions, utility cuts, permits
│   ├── simulations/        # Event simulation HTML maps + results
│   ├── monitor_state/      # Delta detection state between runs
│   ├── vlm_results/        # Camera analysis JSON
│   └── commute_history/    # Commute recommendation log
└── models/
    ├── xgb_congestion_binary.json
    ├── xgb_congestion_multi.json
    └── xgb_volume_regression.json
```

## Setup

```bash
# On MacBook (data prep + training + simulation)
pip install pandas numpy scikit-learn xgboost pyarrow folium requests

# On DGX Spark (add VLM + GPU acceleration)
pip install cudf-cu12 cuml-cu12    # RAPIDS for GPU data processing
ollama pull gemma3:4b               # Vision model for camera analysis
```

## Usage

### 1. Prepare data and train

```bash
python3 scripts/01_prepare_traffic_data.py   # Pull 346K records from CKAN
python3 scripts/02_train_model.py            # Train 3 XGBoost models (~10s)
python3 scripts/08_enrich_traffic_data.py    # Pull TTC delays, collisions, permits
```

### 2. Simulate events

```bash
# Raptors game at Scotiabank Arena
python3 scripts/10_simulate_events.py \
  --event sports_game --location "scotiabank arena" \
  --crowd 20000 --day friday --hour 19

# Live mode — pull real events from Toronto Open Data
python3 scripts/10_simulate_events.py --live --date 2026-07-01

# Gardiner Expressway closure
python3 scripts/10_simulate_events.py \
  --event road_closure --lat 43.636 --lon -79.400 \
  --day monday --hour 8
```

Opens an interactive HTML map showing impact radius, affected intersections, and before/after congestion levels. Live mode pulls real special events and road restrictions from CKAN.

### 3. Route between two points

```bash
python3 scripts/11_route_advisor.py --from "King and Spadina" --to "Finch and Yonge"
python3 scripts/11_route_advisor.py --from "downtown" --to "airport"
```

### 4. Optimize your commute

```bash
# Save your commute (one time)
cat > ~/.commute.json << 'EOF'
{
  "home": {"name": "Liberty Village", "lat": 43.638, "lon": -79.420, "desc": "Liberty Village"},
  "work": {"name": "North York", "lat": 43.762, "lon": -79.411, "desc": "North York Centre"},
  "arrive_by": "09:00"
}
EOF

# Daily morning advice
python3 scripts/12_commute_optimizer.py

# Evening reverse commute
python3 scripts/12_commute_optimizer.py --reverse

# Check specific departure window
python3 scripts/12_commute_optimizer.py --window 7:00-9:30 --detailed
```

### 5. Live monitoring (requires DGX Spark + Ollama)

```bash
# Quick check — 25 cameras, alert on changes
python3 scripts/07_hermes_traffic_monitor.py --mode alert --cameras 25

# Full sweep with Telegram-formatted summary
python3 scripts/07_hermes_traffic_monitor.py --mode summary --cameras 50

# Actionable advice with route recommendations
python3 scripts/09_hermes_actionable_monitor.py --cameras 50
```

### 6. VLM feedback loop + nowcasting

```bash
# Consolidate VLM history + retrain with VLM features
python3 scripts/14_vlm_feedback_loop.py

# Compare base vs VLM-enhanced model accuracy
python3 scripts/14_vlm_feedback_loop.py --compare

# Predict next-hour congestion from current VLM state
python3 scripts/14_vlm_feedback_loop.py --nowcast
```

### 7. Live VLM feed (demo or production)

```bash
# Demo mode — synthetic data, no GPU needed, perfect for presentations
python3 scripts/15_vlm_orchestrator.py --demo --interval 60

# Live mode on DGX Spark — real camera analysis via Ollama
python3 scripts/15_vlm_orchestrator.py --live --cameras 50 --interval 300

# Quick demo with fast cycles (30s between sweeps)
python3 scripts/15_vlm_orchestrator.py --demo --interval 30 --cameras 15

# Run 5 cycles and stop
python3 scripts/15_vlm_orchestrator.py --demo --interval 60 --cycles 5
```

Open the unified dashboard alongside to see live updates:
```bash
streamlit run dashboard.py  # Set auto-refresh to 15s or 30s
```

### 8. Spatio-temporal GNN forecaster

A graph neural network that models how congestion **propagates across the road network** — the one capability XGBoost structurally lacks — and emits a multi-horizon forecast that feeds the dashboard's Nowcast tab. Pure PyTorch (no `torch_geometric` dependency), trains in seconds on `device="cuda"`.

```bash
# Train on the Spark GPU and emit an initial forecast
python3 scripts/16_gnn_forecast.py --train --epochs 80

# Produce a multi-horizon forecast from the current network state
python3 scripts/16_gnn_forecast.py --forecast

# Demo forecast — synthetic, no torch/data, for presentations
python3 scripts/16_gnn_forecast.py --demo
```

Architecture: self-adaptive adjacency `A = softmax(relu(E1·E2ᵀ))` (graph is learned, no lat/lon needed) → GRU temporal encoder → 2× diffusion graph convolution → linear decoder. Trained with sensor dropout so masked intersections are recovered from their neighbours. See the **Architecture** tab in the dashboard and the Appendix in the top-level README for the full write-up.

### 9. Hermes Agent integration

Ask Hermes to schedule automatic monitoring:

> "Every weekday at 6:45 AM, run the commute optimizer and send me the result on Telegram"

> "Check Toronto traffic every 15 minutes and alert me if any major road hits gridlock"

## Data Sources

### Primary
- **[Turning Movement Counts](https://open.toronto.ca/dataset/traffic-volumes-at-intersections-for-all-modes/)** — 346K records at 4,058 intersections (2020-2026), 15-min intervals with per-direction vehicle/ped/bike counts
- **[Traffic Cameras](https://open.toronto.ca/dataset/traffic-cameras/)** — 336 live camera feeds with lat/lon, updated every few minutes

### Enrichment
- **[TTC Subway Delays](https://open.toronto.ca/dataset/ttc-subway-delay-data/)** — 35K records, line/station/cause/duration
- **[TTC Bus Delays](https://open.toronto.ca/dataset/ttc-bus-delay-data/)** — 50K records with route-level detail
- **[TTC Streetcar Delays](https://open.toronto.ca/dataset/ttc-streetcar-delay-data/)** — 19K records
- **[Motor Vehicle Collisions (KSI)](https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/)** — 20K records (2006-2026) with lat/lon
- **[Utility Cut Permits](https://open.toronto.ca/dataset/utility-cut/)** — 88K active road excavation permits
- **[Building Permits](https://open.toronto.ca/dataset/building-permits-active-permits/)** — 50K construction permits with dates
- **[Liquor Licence Special Events](https://open.toronto.ca/dataset/liquor-licence-special-events/)** — 4,420 municipally significant events with lat/lon, dates, crowd estimates
- **[Road Restrictions](https://secure.toronto.ca/opendata/cart/road_restrictions/)** — Live feed of 2,700+ active road closures and restrictions

## Feature Engineering (30 features)

- **Temporal** (14): hour, DOW, month, quarter, cyclical sin/cos encodings, rush hour flags, weekend
- **Volume** (7): total vehicles/peds/bikes, modal split percentages, vehicle-to-ped ratio
- **Location** (9): per-intersection mean/std/max vehicles, mean peds/bikes, record count, hour-location and DOW-location interaction means, encoded location ID

## Hardware

- **MacBook Pro**: data prep, model training, simulation, routing — all run in seconds on CPU
- **DGX Spark (GB10 Blackwell)**: VLM camera analysis (gemma3:4b, ~2s/frame), GPU-accelerated training via RAPIDS/cuML, 24/7 Hermes Agent monitoring with Telegram delivery
