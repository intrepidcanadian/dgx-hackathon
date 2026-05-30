# Toronto Open Data + DGX Spark Projects

Three ML projects built on Toronto Open Data, designed to run on NVIDIA DGX Spark (GB10 Blackwell GPU, 128GB unified memory). Data prep and model training run on a MacBook Pro; VLM inference, GPU-accelerated training, and 24/7 monitoring deploy to the Spark.

## Projects

### [Traffic Intelligence Platform](traffic/)
Real-time congestion prediction, event simulation, and commute optimization using 346K turning movement counts and 336 live traffic cameras. XGBoost prediction (0.99 AUC) combined with Ollama vision-language models for live camera analysis.

**13 scripts** — data prep, GPU training, VLM camera analysis, Streamlit dashboard, Hermes monitoring with Telegram alerts, event injection simulation with interactive maps, point-to-point routing, daily commute optimization, and live event enrichment from Toronto Open Data.

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
