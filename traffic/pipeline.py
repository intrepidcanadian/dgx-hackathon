#!/usr/bin/env python3
"""
Toronto Traffic Intelligence — unified pipeline entry point.

One file to run the whole thing. Instead of remembering which of the 16
numbered scripts to run in what order, call this with a stage:

    python3 pipeline.py build              # full offline build (data → models)
    python3 pipeline.py build --gpu        # use RAPIDS/cuML GPU variants on the Spark
    python3 pipeline.py data               # pull + engineer + enrich data only
    python3 pipeline.py train              # train XGBoost + VLM nowcast + GNN
    python3 pipeline.py vlm --demo         # live VLM camera feed (synthetic, no GPU)
    python3 pipeline.py vlm --live --cameras 50 --interval 300   # real sweeps on Spark
    python3 pipeline.py nowcast            # recompute next-hour nowcast from current VLM state
    python3 pipeline.py forecast           # emit GNN multi-horizon forecast
    python3 pipeline.py dashboard          # launch the Streamlit dashboard
    python3 pipeline.py all                # build + dashboard

Each stage shells out to the corresponding numbered script in ./scripts so
their individual behaviour, flags, and outputs are preserved exactly. This
file is the *interface* — it does not duplicate the modelling logic.

The full build sequence mirrors deploy_spark.sh:
    01 prepare → 08 enrich → 13 events → 02 train → 14 vlm nowcast → 16 gnn
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
DASHBOARD = HERE.parent / "dashboard.py"
DATA_DIRS = [
    "data/raw", "data/processed", "data/enrichment", "data/simulations",
    "data/monitor_state", "data/vlm_results", "data/commute_history", "models",
]


# ---------------------------------------------------------------- helpers ----
def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""


BOLD, GREEN, RED, YEL, CYAN, RST = (
    _c("\033[1m"), _c("\033[32m"), _c("\033[31m"),
    _c("\033[33m"), _c("\033[36m"), _c("\033[0m"),
)


def step(msg: str):
    print(f"\n{BOLD}{CYAN}▶ {msg}{RST}", flush=True)


def ok(msg: str):
    print(f"{GREEN}✓ {msg}{RST}", flush=True)


def warn(msg: str):
    print(f"{YEL}⚠ {msg}{RST}", flush=True)


def fail(msg: str):
    print(f"{RED}✗ {msg}{RST}", flush=True)


def have_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def ollama_up(url: str = "http://localhost:11434") -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def resolve_script(stem: str, gpu: bool) -> Path:
    """Return the script path, preferring the _gpu variant when --gpu and it exists."""
    if gpu:
        gpu_variant = SCRIPTS / f"{stem}_gpu.py"
        if gpu_variant.exists():
            return gpu_variant
    return SCRIPTS / f"{stem}.py"


def run(stem: str, *args: str, gpu: bool = False, check: bool = True,
        cont: bool = False) -> int:
    """Run a stage script via subprocess, streaming its output live."""
    script = resolve_script(stem, gpu)
    if not script.exists():
        fail(f"Script not found: {script.name}")
        if check and not cont:
            sys.exit(1)
        return 127
    cmd = [sys.executable, str(script), *args]
    label = " ".join([script.name, *args])
    print(f"  {BOLD}$ {label}{RST}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    if proc.returncode == 0:
        ok(f"{script.name} done ({dt:.1f}s)")
    else:
        fail(f"{script.name} exited {proc.returncode} ({dt:.1f}s)")
        if check and not cont:
            sys.exit(proc.returncode)
    return proc.returncode


def ensure_dirs():
    for d in DATA_DIRS:
        (HERE / d).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- stages ----
def stage_data(gpu: bool, cont: bool):
    step("Data — pull from CKAN, engineer features, enrich")
    run("01_prepare_traffic_data", gpu=gpu, cont=cont)
    # Per-camera measured baseline from City midblock count stations — the
    # ground truth each live VLM read is scored against (busier/quieter than
    # typical + typical speed). Produces data/processed/camera_baseline.parquet.
    run("17_camera_baseline", cont=cont)
    run("08_enrich_traffic_data", cont=cont)
    run("13_enrich_events", cont=cont)
    # Join historical hourly weather (Open-Meteo ERA5) onto train/test so the
    # Analytics "Weather Impact" panel has real weather columns to correlate.
    run("18_enrich_weather", cont=cont)


def stage_train(gpu: bool, epochs: int, cont: bool):
    step("Train — XGBoost congestion, VLM nowcast, spatio-temporal GNN")
    run("02_train_model", gpu=gpu, cont=cont)
    # Leakage-free VLM nowcast (observe t -> predict t+1); --compare prints metrics
    run("14_vlm_feedback_loop", "--compare", cont=cont)
    # GNN: train then emit an initial forecast
    run("16_gnn_forecast", "--train", "--epochs", str(epochs), cont=cont)
    run("16_gnn_forecast", "--forecast", cont=cont)


def stage_vlm(args):
    step("VLM — live traffic-camera analysis (gemma3 via Ollama)")
    if not args.demo and not ollama_up(args.ollama_url):
        warn(f"Ollama not reachable at {args.ollama_url}. "
             "Start it (`ollama serve` + `ollama pull gemma3:4b`) or use --demo.")
    extra = []
    if args.demo:
        extra.append("--demo")
    else:
        extra.append("--live")
    extra += ["--cameras", str(args.cameras), "--interval", str(args.interval),
              "--ollama-url", args.ollama_url, "--model", args.model]
    if args.cycles:
        extra += ["--cycles", str(args.cycles)]
    if getattr(args, "save_frames", False):
        extra.append("--save-frames")
    run("15_vlm_orchestrator", *extra)


def stage_nowcast(cont: bool):
    step("Nowcast — next-hour congestion from current VLM state")
    run("14_vlm_feedback_loop", "--nowcast", cont=cont)


def stage_forecast(cont: bool):
    step("Forecast — GNN multi-horizon network forecast")
    run("16_gnn_forecast", "--forecast", cont=cont)


def stage_dashboard():
    step("Dashboard — launching Streamlit")
    if not DASHBOARD.exists():
        fail(f"dashboard.py not found at {DASHBOARD}")
        sys.exit(1)
    print(f"  {BOLD}$ streamlit run {DASHBOARD.name}{RST}", flush=True)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(DASHBOARD)],
                   cwd=DASHBOARD.parent)


def stage_build(gpu: bool, epochs: int, cont: bool):
    t0 = time.time()
    ensure_dirs()
    stage_data(gpu, cont)
    stage_train(gpu, epochs, cont)
    step("Build complete")
    ok(f"All models + data ready in {time.time() - t0:.1f}s. "
       "Run `python3 pipeline.py dashboard` to view.")


# ------------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser(
        description="Unified Toronto Traffic Intelligence pipeline. "
                    "Drives the numbered ./scripts stages in the right order. "
                    "Common: `pipeline.py build` then `pipeline.py dashboard`.",
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def add_gpu(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--gpu", action="store_true",
                       help="prefer RAPIDS/cuML _gpu script variants")
        g.add_argument("--no-gpu", action="store_true",
                       help="force CPU scripts even if a GPU is present")
        sp.add_argument("--continue-on-error", dest="cont", action="store_true",
                        help="keep going if a stage fails")

    sp_build = sub.add_parser("build", help="full offline build (data → models)")
    add_gpu(sp_build)
    sp_build.add_argument("--epochs", type=int, default=80, help="GNN training epochs")

    sp_all = sub.add_parser("all", help="build, then launch the dashboard")
    add_gpu(sp_all)
    sp_all.add_argument("--epochs", type=int, default=80)

    sp_data = sub.add_parser("data", help="pull + engineer + enrich data only")
    add_gpu(sp_data)

    sp_train = sub.add_parser("train", help="train XGBoost + VLM nowcast + GNN")
    add_gpu(sp_train)
    sp_train.add_argument("--epochs", type=int, default=80)

    sp_vlm = sub.add_parser("vlm", help="live VLM camera feed (orchestrator)")
    mode = sp_vlm.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="synthetic, no GPU/Ollama")
    mode.add_argument("--live", action="store_true", help="real camera sweeps (default)")
    sp_vlm.add_argument("--cameras", type=int, default=50)
    sp_vlm.add_argument("--interval", type=int, default=300, help="seconds between sweeps")
    sp_vlm.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    sp_vlm.add_argument("--save-frames", dest="save_frames", action="store_true",
                        help="archive each analyzed JPEG to data/camera_frames/")
    sp_vlm.add_argument("--ollama-url", default="http://localhost:11434")
    sp_vlm.add_argument("--model", default="gemma3:4b")

    sub.add_parser("nowcast", help="recompute next-hour nowcast from VLM state")
    sub.add_parser("forecast", help="emit GNN multi-horizon forecast")
    sub.add_parser("dashboard", help="launch the Streamlit dashboard")

    args = p.parse_args()

    # Resolve effective GPU preference
    gpu = False
    if getattr(args, "gpu", False):
        if not have_gpu():
            warn("--gpu requested but nvidia-smi not found; using CPU scripts.")
        else:
            gpu = True
    elif getattr(args, "no_gpu", False):
        gpu = False

    cont = getattr(args, "cont", False)

    if args.stage == "build":
        stage_build(gpu, args.epochs, cont)
    elif args.stage == "all":
        stage_build(gpu, args.epochs, cont)
        stage_dashboard()
    elif args.stage == "data":
        ensure_dirs()
        stage_data(gpu, cont)
    elif args.stage == "train":
        stage_train(gpu, args.epochs, cont)
    elif args.stage == "vlm":
        stage_vlm(args)
    elif args.stage == "nowcast":
        stage_nowcast(cont)
    elif args.stage == "forecast":
        stage_forecast(cont)
    elif args.stage == "dashboard":
        stage_dashboard()


if __name__ == "__main__":
    main()
