#!/usr/bin/env bash
# ============================================================
# DGX Spark Deployment Script
# ============================================================
#
# Deploys all three Toronto Open Data ML projects to an NVIDIA
# DGX Spark (GB10 Blackwell GPU, 128GB unified memory).
#
# What this script does:
#   1. Installs Python dependencies
#   2. Installs Ollama and pulls required models
#   3. Regenerates data + trains models (data is gitignored)
#   4. Validates everything works
#   5. Prints Hermes Agent setup instructions (interactive)
#
# Prerequisites:
#   - DGX Spark running DGX OS (Ubuntu-based)
#   - This repo cloned:  git clone <repo> ~/dgx && cd ~/dgx
#   - Internet access for CKAN API + Ollama model downloads
#
# Usage:
#   bash deploy_spark.sh           # Full deploy (all projects)
#   bash deploy_spark.sh traffic   # Traffic project only
#   bash deploy_spark.sh dinesafe  # DineSafe project only
#   bash deploy_spark.sh housing   # Housing project only
#   bash deploy_spark.sh hermes    # Hermes setup only
#
# Time estimate:
#   ~5 min for deps + data + training
#   ~20 min for Ollama model downloads (depends on network)
#   ~30 min for Hermes interactive setup
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; }

TARGET="${1:-all}"

# ============================================================
# STEP 1: System checks
# ============================================================
step "1/6  System checks"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "nvidia-smi not found — GPU features will be unavailable"
fi

python3 --version || { fail "Python3 not found"; exit 1; }
ok "Python3 available"

# ============================================================
# STEP 2: Python dependencies
# ============================================================
step "2/6  Python dependencies"

# DGX OS uses externally-managed Python (PEP 668) — must use a venv
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment exists"
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate"
ok "Activated venv: $(which python3)"

pip install --quiet --upgrade pip

# Core deps (all projects)
pip install --quiet \
    pandas numpy scikit-learn xgboost pyarrow requests \
    matplotlib folium streamlit torch

# GPU deps (DGX Spark specific — skip if not available)
if command -v nvidia-smi &>/dev/null; then
    echo "Installing RAPIDS (cuDF/cuML) for GPU acceleration..."
    pip install --quiet cudf-cu12 cuml-cu12 2>/dev/null && \
        ok "RAPIDS installed" || \
        warn "RAPIDS install failed — CPU fallback will be used"
fi

ok "Python dependencies installed"

# ============================================================
# STEP 3: Ollama + models
# ============================================================
step "3/6  Ollama + models"

if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed"
fi

# Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama service..."
    sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
    sleep 3
fi

# Pull models
echo "Pulling gemma3:4b (VLM for traffic camera analysis)..."
ollama pull gemma3:4b
ok "gemma3:4b ready"

echo "Pulling nemotron-3-super (NLP features + dashboard AI summaries)..."
ollama pull nemotron-3-super
ok "nemotron-3-super ready"

# ============================================================
# STEP 4: Data + models (per project)
# ============================================================
step "4/6  Data download + model training"

# --- Traffic ---
if [[ "$TARGET" == "all" || "$TARGET" == "traffic" ]]; then
    echo ""
    echo "=== TRAFFIC ==="

    cd "$SCRIPT_DIR/traffic"

    # Single source of truth: pipeline.py drives the full data→models build
    # (01 prepare → 08 enrich → 13 events → 02 train → 14 vlm nowcast → 16 gnn).
    # Uses the RAPIDS/cuML _gpu script variants when a GPU is present.
    if command -v nvidia-smi &>/dev/null; then
        python3 pipeline.py build --gpu
    else
        python3 pipeline.py build
    fi
    ok "Traffic data + models built (pipeline.py)"
fi

# --- DineSafe ---
if [[ "$TARGET" == "all" || "$TARGET" == "dinesafe" ]]; then
    echo ""
    echo "=== DINESAFE ==="

    cd "$SCRIPT_DIR/dinesafe"
    mkdir -p data/raw data/processed data/kg_documents/markdown models

    echo "  Downloading DineSafe data..."
    bash scripts/01_download_data.sh
    ok "DineSafe data downloaded"

    echo "  Preparing features..."
    python3 scripts/03_prepare_features.py
    ok "Features prepared (31 base)"

    echo "  Enriching features..."
    python3 scripts/05_enrich_features.py
    python3 scripts/06_enrich_v2.py
    ok "Features enriched (58 total)"

    echo "  Training model..."
    python3 scripts/04_train_model.py
    ok "DineSafe model trained"

    echo "  Extracting NLP features via Nemotron..."
    python3 scripts/07_nemotron_features.py
    ok "NLP features extracted"

    echo "  Training improved model..."
    python3 scripts/08_improve_model.py
    ok "Improved model trained"
fi

# --- Housing ---
if [[ "$TARGET" == "all" || "$TARGET" == "housing" ]]; then
    echo ""
    echo "=== HOUSING ==="

    cd "$SCRIPT_DIR/housing"
    mkdir -p data/raw data/processed models

    echo "  Downloading shelter + weather data..."
    bash scripts/01_download_data.sh
    ok "Housing data downloaded"

    echo "  Preparing features..."
    python3 scripts/03_prepare_features.py
    ok "Features prepared (52 features)"

    echo "  Training XGBoost..."
    python3 scripts/04_train_model.py
    ok "XGBoost trained"

    echo "  Training LSTM..."
    python3 scripts/05_train_lstm.py
    ok "LSTM trained"

    echo "  Enriching features..."
    python3 scripts/10_enrich_features.py
    ok "Enrichment data added"
fi

cd "$SCRIPT_DIR"

# ============================================================
# STEP 5: Validation
# ============================================================
step "5/6  Validation"

PASS=0
TOTAL=0

validate() {
    TOTAL=$((TOTAL + 1))
    if [ -f "$1" ]; then
        ok "$2"
        PASS=$((PASS + 1))
    else
        fail "$2 — missing: $1"
    fi
}

if [[ "$TARGET" == "all" || "$TARGET" == "traffic" ]]; then
    validate "traffic/models/xgb_congestion_binary.json" "Traffic binary model"
    validate "traffic/models/xgb_congestion_multi.json"  "Traffic multi-class model"
    validate "traffic/models/xgb_volume_regression.json"  "Traffic volume model"
    validate "traffic/data/processed/test.parquet"        "Traffic test data"
    validate "traffic/data/processed/feature_cols.json"   "Traffic feature columns"
    validate "traffic/data/raw/traffic_cameras.csv"       "Traffic cameras"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "dinesafe" ]]; then
    validate "dinesafe/models/xgb_risk_final.json"        "DineSafe model"
    validate "dinesafe/data/processed/train.parquet"       "DineSafe train data"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "housing" ]]; then
    validate "housing/data/processed/test.parquet"         "Housing test data"
fi

echo ""
echo "$PASS/$TOTAL validations passed"

# Quick smoke test — commute optimizer
if [[ "$TARGET" == "all" || "$TARGET" == "traffic" ]]; then
    echo ""
    echo "Smoke test: commute optimizer..."
    python3 traffic/scripts/12_commute_optimizer.py \
        --from "downtown" --to "north york" \
        --simulate-day tuesday 2>/dev/null | head -5
    ok "Commute optimizer working"
fi

# ============================================================
# STEP 6: Hermes Agent setup
# ============================================================
step "6/6  Hermes Agent setup"

if [[ "$TARGET" == "all" || "$TARGET" == "hermes" ]]; then
    if command -v hermes &>/dev/null; then
        ok "Hermes already installed"
        echo ""
        echo "To update: hermes update"
    else
        echo "Hermes Agent needs interactive setup (~30 min)."
        echo ""
        echo "Install now? This will:"
        echo "  1. Install Hermes Agent"
        echo "  2. Connect to Ollama (localhost:11434)"
        echo "  3. Set up Telegram bot (you'll need a bot token from @BotFather)"
        echo "  4. Install gateway as system service (survives reboots)"
        echo ""
        read -p "Install Hermes now? [Y/n] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z "$REPLY" ]]; then
            curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
        else
            echo ""
            echo "To install later:"
            echo "  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"
        fi
    fi

    echo ""
    echo -e "${BLUE}━━━ HERMES SCHEDULING ━━━${NC}"
    echo ""
    echo "Once Hermes is running, tell it to schedule your traffic monitors."
    echo "Open Hermes (terminal or Telegram) and send these messages:"
    echo ""
    VENV_PY="$SCRIPT_DIR/venv/bin/python3"
    echo -e "${GREEN}Morning commute (weekdays 6:45 AM):${NC}"
    echo '  "Every weekday at 6:45 AM, run this command and send me the output:'
    echo "   $VENV_PY $SCRIPT_DIR/traffic/scripts/12_commute_optimizer.py\""
    echo ""
    echo -e "${GREEN}Evening commute (weekdays 4:30 PM):${NC}"
    echo '  "Every weekday at 4:30 PM, run this command and send me the output:'
    echo "   $VENV_PY $SCRIPT_DIR/traffic/scripts/12_commute_optimizer.py --reverse\""
    echo ""
    echo -e "${GREEN}Traffic monitoring (every 15 min):${NC}"
    echo '  "Every 15 minutes, run this command and alert me if any road hits gridlock:'
    echo "   $VENV_PY $SCRIPT_DIR/traffic/scripts/09_hermes_actionable_monitor.py --cameras 50\""
    echo ""
    echo -e "${GREEN}Full camera sweep (every hour):${NC}"
    echo '  "Every hour from 7 AM to 8 PM on weekdays, run:'
    echo "   $VENV_PY $SCRIPT_DIR/traffic/scripts/07_hermes_traffic_monitor.py --mode summary --cameras 100\""
    echo ""

    # Create commute config if it doesn't exist
    COMMUTE_CFG="$HOME/.commute.json"
    if [ ! -f "$COMMUTE_CFG" ]; then
        echo -e "${YELLOW}No commute config found. Creating default (~/.commute.json)...${NC}"
        echo "Edit this file with your actual home/work locations."
        echo ""
        cat > "$COMMUTE_CFG" << 'COMMUTE_EOF'
{
  "home": {"name": "Liberty Village", "lat": 43.6380, "lon": -79.4195, "desc": "Liberty Village"},
  "work": {"name": "North York Centre", "lat": 43.7615, "lon": -79.4111, "desc": "North York Centre"},
  "arrive_by": "09:00"
}
COMMUTE_EOF
        ok "Created $COMMUTE_CFG — edit with your locations"
    else
        ok "Commute config exists: $COMMUTE_CFG"
    fi
fi

# ============================================================
# DONE
# ============================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Deployment complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "What's running:"
echo "  • Ollama serving gemma3:4b on localhost:11434"
if command -v hermes &>/dev/null; then
echo "  • Hermes Agent with Telegram gateway (systemd service)"
fi
echo ""
echo "Quick commands (activate venv first: source venv/bin/activate):"
echo "  streamlit run traffic/scripts/04_dashboard.py       # Traffic dashboard"
echo "  streamlit run dinesafe/scripts/05_dashboard.py      # DineSafe dashboard"
echo "  streamlit run housing/scripts/06_dashboard.py       # Housing dashboard"
echo "  python3 traffic/scripts/12_commute_optimizer.py     # Morning commute"
echo "  python3 traffic/scripts/10_simulate_events.py --live # Live event map"
echo ""
