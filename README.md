# Toronto Open Data + DGX Spark Projects

Three ML projects built on Toronto Open Data, designed to run on NVIDIA DGX Spark (GB10 Blackwell GPU, 128GB unified memory). Data prep and model training run on a MacBook Pro; VLM inference, GPU-accelerated training, and 24/7 monitoring deploy to the Spark.

## Projects

### [Traffic Intelligence Platform](traffic/)
Real-time congestion prediction, event simulation, and commute optimization using 346K turning movement counts and 336 live traffic cameras. XGBoost prediction (0.99 AUC) combined with Ollama vision-language models for live camera analysis.

**16 scripts** — data prep, GPU training, VLM camera analysis, Streamlit dashboard, Hermes monitoring with Telegram alerts, event injection simulation with interactive maps, point-to-point routing, daily commute optimization, live event enrichment, VLM feedback loop with nowcasting, continuous VLM orchestrator for live demos, and a spatio-temporal GNN multi-horizon forecaster.

### [DineSafe Risk Predictor](dinesafe/)
Predict restaurant health inspection failures using violation history, 311 complaints, fire incidents, and Nemotron NLP features. Neighbourhood-level knowledge graph via txt2kg for cross-domain insight.

**11 scripts** — data prep, enrichment, XGBoost training (0.82 AUC), NLP feature extraction, Streamlit dashboards, and knowledge graph ingestion.

### [Shelter Demand Predictor](housing/)
Forecast shelter occupancy 1 day ahead and detect emerging homelessness risk by ward using shelter data, weather, bike share signals, and RentSafeTO building evaluations. Three model architectures: XGBoost (0.93 AUC), LSTM, and Temporal Fusion Transformer.

**11 scripts** — data prep, three GPU model trainers, Streamlit dashboard, hourly weather enrichment, and ward-level early warning system.

## Common Stack

| Component | Purpose |
|-----------|---------|
| **Toronto Open Data (CKAN API)** | All primary datasets — inspections, traffic, shelter occupancy, permits, TTC delays, collisions |
| **Environment Canada** | Daily and hourly weather for shelter + traffic models |
| **XGBoost** | Primary ML model across all three projects, `device="cuda"` on Spark |
| **Ollama** | VLM camera analysis (gemma3), NLP features (nemotron), knowledge graph triples |
| **RAPIDS (cuDF/cuML)** | GPU-accelerated pandas and scikit-learn on Spark |
| **Folium** | Interactive map visualization for traffic simulation |
| **Streamlit** | Dashboards for all three projects |
| **Hermes Agent** | 24/7 scheduled monitoring with Telegram delivery |
| **txt2kg** | Knowledge graph construction from neighbourhood documents |

## Hardware

| Task | MacBook Pro | DGX Spark |
|------|-------------|-----------|
| Data prep (CKAN pulls, feature engineering) | seconds | seconds (or GPU via RAPIDS) |
| XGBoost training (300K rows) | ~10s CPU | ~2s `device="cuda"` |
| VLM camera analysis (336 cameras) | N/A | ~11 min (gemma3:4b) |
| LSTM / TFT training | minutes CPU | seconds GPU |
| Hermes 24/7 monitoring + Telegram | laptop must stay open | always-on systemd service |
| txt2kg knowledge graph | N/A | local LLM triple extraction |

## Deploy to DGX Spark

One script handles everything — deps, Ollama, data, training, and Hermes Agent setup:

```bash
# On DGX Spark
git clone <your-repo> ~/dgx && cd ~/dgx
bash deploy_spark.sh              # Full deploy (~30 min with Hermes)
```

Or deploy individual projects:

```bash
bash deploy_spark.sh traffic      # Traffic only (~2 min)
bash deploy_spark.sh dinesafe     # DineSafe only (~3 min)
bash deploy_spark.sh housing      # Housing only (~2 min)
bash deploy_spark.sh hermes       # Hermes Agent only (~30 min)
```

### What the deploy script does

1. **System checks** — verifies GPU, Python3
2. **Python deps** — installs all packages + RAPIDS for GPU acceleration
3. **Ollama** — installs Ollama, pulls gemma3:4b (VLM) and nemotron (NLP)
4. **Data + models** — pulls from CKAN APIs, trains all XGBoost/LSTM models
5. **Validation** — checks all model files and data artifacts exist
6. **Hermes Agent** — interactive install with Telegram bot setup

### Hermes scheduled tasks

Once Hermes is running, tell it (via terminal or Telegram):

- *"Every weekday at 6:45 AM, run the commute optimizer and send me the output"*
- *"Every weekday at 4:30 PM, run the commute optimizer with --reverse"*
- *"Every 15 minutes, run the traffic monitor and alert me if any road hits gridlock"*

### Why data isn't in git

Data (`data/raw/`, `data/processed/`) and models (`models/`) are gitignored. The deploy script regenerates everything from Toronto Open Data's CKAN API in under a minute — and you get the freshest data on the Spark.

## Local Setup (MacBook)

```bash
pip install pandas numpy scikit-learn xgboost pyarrow requests folium streamlit torch
```

Run scripts 1–7 in the traffic project (and equivalents in dinesafe/housing) for data prep, training, simulation, and routing. Only VLM camera analysis and 24/7 Hermes monitoring require the Spark.

See each project's README for detailed usage instructions.

---

# Appendix: Architecture & Modeling

This appendix documents how each project models its data, which model
runs where, and exactly which workloads use the DGX Spark GPU.

## A.1 — Design principle: trees for tabular, GPU for the rest

The core prediction tasks are **tabular** (tens of engineered features,
hundreds of thousands of rows). On data of this shape, gradient-boosted
trees are at or near state-of-the-art, so **XGBoost is the backbone of
all three projects**. We don't replace it with deep nets to chase
accuracy — there's little headroom (the traffic binary classifier is at
0.9937 AUC).

The GPU is reserved for workloads that trees **structurally cannot do**:

| Workload | Model | Why it needs the GPU / why not XGBoost |
|----------|-------|-----------------------------------------|
| Live camera understanding | gemma3:4b VLM (Ollama) | Vision-language inference over 336 camera frames; no tabular equivalent |
| Congestion **propagation** forecasting | Spatio-temporal GNN (PyTorch) | Trees score each intersection independently and can't model the road network as a graph |
| Sequence forecasting w/ uncertainty | LSTM, Temporal Fusion Transformer (PyTorch) | Multi-horizon shelter occupancy with quantile bands |
| NLP feature extraction | nemotron (Ollama) | Free-text inspection notes / complaints → structured features |
| Knowledge graph construction | txt2kg + local LLM | Triple extraction from neighbourhood documents |
| Accelerated tabular training | XGBoost `device="cuda"`, RAPIDS cuDF/cuML | Same model, ~5× faster data prep + training |

## A.2 — Traffic: data modeling

**Target.** Per-intersection `congestion_level` ∈ {0,1,2,3}, derived by
per-location quartile binning of 15-minute vehicle volume (so a level is
comparable across very different intersections). A binary `is_congested`
(top quartile) and continuous `total_traffic` regression target are also
produced.

**Features (30).** Temporal (hour, day-of-week, month, cyclical sin/cos
encodings, rush-hour flags), volume/modal-split (vehicle/ped/bike totals
and ratios), and location-level history (per-intersection mean/std/max,
hour×location and DOW×location interaction means, encoded location ID).
See `traffic/scripts/01_prepare_traffic_data.py`.

**Models.**
- **XGBoost** — binary (0.9937 AUC), 4-class (87%), and volume
  regression (MAE 4.6). `device="cuda"` on the Spark.
- **VLM-enhanced XGBoost** (`14_vlm_feedback_loop.py`) — adds live VLM
  camera observations as features, then nowcasts next-hour congestion.
- **Spatio-temporal GNN** (`16_gnn_forecast.py`, *new*) — see A.3.

## A.3 — The spatio-temporal GNN forecaster (`16_gnn_forecast.py`)

The one place a neural net adds a capability XGBoost cannot: modeling how
congestion **spreads across the road network** over time, and producing a
**multi-horizon** forecast for the dashboard timeline.

**Architecture** (compact Graph-WaveNet style, pure PyTorch — no
`torch_geometric` dependency so it deploys cleanly on Blackwell/ARM):

```
node embeddings  ->  self-adaptive adjacency  A = softmax(relu(E1 @ E2ᵀ))
GRU temporal encoder (per node)               ->  H  [B, N, hidden]
2× diffusion graph convolution  H' = A @ H     ->  spatial message passing
linear decoder                                ->  next T_out steps
```

- **Self-adaptive adjacency** — the network graph is *learned* from data
  (node embeddings), so no lat/lon or hand-built road graph is required.
- **Data representation** — counts are resampled into the network's
  typical weekly cycle (7×24 = 168 steps) of per-location congestion;
  sliding windows over that axis are the training examples. With a live
  continuous feed you swap the cyclic series for a rolling buffer — the
  model code is unchanged.
- **Sensor dropout** — during training, random nodes' inputs are masked
  so the model must reconstruct them from network neighbours, forcing the
  adjacency to learn real spatial structure.
- **GPU** — trains with `device="cuda"`; ~26K params, seconds on the
  Spark. `--demo` produces a synthetic forecast with no torch/data for
  presentations.

**Honest evaluation.** On the smooth weekly profile, the STGNN beats a
persistence baseline on the **multi-horizon** task (the metric that
matters — persistence just repeats a flat value and can't show the
rush-hour rise/fall the timeline needs). Sensor-down recovery is a
secondary diagnostic. We did **not** swap XGBoost for the GNN anywhere
the goal is single-point accuracy.

## A.4 — DineSafe & Housing: data modeling

- **DineSafe** — XGBoost (0.82 AUC) over inspection history + 311
  complaints + fire incidents, with **nemotron** NLP features from
  free-text notes and a **txt2kg** neighbourhood knowledge graph for
  cross-domain signal.
- **Housing** — three architectures on shelter occupancy + weather +
  bike-share + RentSafeTO: **XGBoost** (0.93 AUC) for the point forecast,
  **LSTM** and **Temporal Fusion Transformer** for sequence/uncertainty.
  LSTM/TFT train on the GPU.

## A.5 — Serving architecture (how models reach the dashboard)

Models don't talk to the dashboard directly — they write small JSON/CSV
**state files** that the Streamlit app polls (15–30s auto-refresh). This
decouples slow GPU inference from the always-fast UI:

```
VLM orchestrator (15) ─┐
VLM feedback loop (14) ─┼─► data/monitor_state/*.json ─► dashboard.py (Streamlit)
GNN forecaster   (16) ─┘     latest_nowcast.json              ├─ Command Center (pydeck maps)
Hermes monitor   (07/09)     latest_forecast.json             ├─ Nowcast + Network Forecast
                             orchestrator_status.json         └─ Event Sim + Commute Planner
```

- **GPU side (Spark):** VLM sweeps, GNN training/forecasting, XGBoost
  `device="cuda"`, Hermes 24/7 monitoring (systemd) with Telegram.
- **UI side:** the dashboard reads state files only — it has no GPU
  dependency and runs anywhere, degrading gracefully when a feed is idle.
