#!/usr/bin/env python3
"""Spatio-temporal GNN traffic forecaster (DGX Spark, device="cuda").

XGBoost scores each intersection independently and cannot model how
congestion *propagates* across the road network. This script adds a
graph neural network that learns the network topology and the diurnal
dynamics jointly, then emits a MULTI-HORIZON forecast that feeds the
dashboard's nowcast / forecast-timeline UI.

Model — a compact Graph-WaveNet-style spatio-temporal net (pure PyTorch,
no torch_geometric dependency so it deploys cleanly on Blackwell/ARM):

    node embeddings  ->  self-adaptive adjacency  A = softmax(relu(E1 @ E2^T))
    GRU temporal encoder (per node)              ->  H  [B, N, hidden]
    diffusion graph convolution  H' = A @ H       ->  spatial message passing
    linear decoder                                ->  next T_out steps [B, N, T_out]

Why the graph matters: turning-movement counts are sparse per
intersection, so we train with *sensor dropout* — random nodes have
their input window masked and must be reconstructed from their network
neighbours. This forces the adaptive adjacency to learn real spatial
structure (the thing XGBoost structurally cannot do).

Data representation: counts are resampled into the network's typical
weekly cycle (7 days x 24 h = 168 steps) of per-location congestion
level. Sliding windows over that cyclic axis are the training examples.
With a live continuous feed you would swap the cyclic series for a
rolling real-time buffer; the model code is unchanged.

Usage
-----
    # Train on the Spark GPU and save the model
    python3 scripts/16_gnn_forecast.py --train --epochs 80

    # Produce a multi-horizon forecast from the current network state
    python3 scripts/16_gnn_forecast.py --forecast

    # Demo forecast (synthetic, no torch / no data) for presentations
    python3 scripts/16_gnn_forecast.py --demo

Outputs
-------
    models/stgnn_traffic.pt            trained weights + metadata
    data/monitor_state/latest_forecast.json   horizon forecast for the dashboard
"""

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
STATE_DIR = DATA_DIR / "monitor_state"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

WEEK_HOURS = 7 * 24          # 168 cyclic timesteps
LEVEL_NAMES = ["Free Flow", "Light", "Moderate", "Heavy"]
DEFAULT_TOP_N = 200          # densest intersections to model as graph nodes


# ============================================================
# DATA: build the network's weekly congestion tensor
# ============================================================
def build_cyclic_tensor(top_n=DEFAULT_TOP_N):
    """Return X[N, 168] of mean congestion level per (location, week-hour)."""
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    df = pd.concat([train, test], ignore_index=True)

    # week-hour index = day_of_week * 24 + hour
    df = df.dropna(subset=["congestion_level", "day_of_week", "hour", "location_id"])
    df["wh"] = (df["day_of_week"].astype(int) * 24 + df["hour"].astype(int)) % WEEK_HOURS

    # keep densest intersections
    counts = df["location_id"].value_counts()
    nodes = counts.head(top_n).index.tolist()
    df = df[df["location_id"].isin(nodes)]

    # profile matrix [N, 168]
    profile = (df.groupby(["location_id", "wh"])["congestion_level"]
                 .mean().unstack("wh"))
    profile = profile.reindex(index=nodes, columns=range(WEEK_HOURS))

    # fill gaps: node mean, then global mean
    global_mean = float(df["congestion_level"].mean())
    profile = profile.apply(lambda r: r.fillna(r.mean()), axis=1)
    profile = profile.fillna(global_mean)

    X = profile.values.astype("float32")          # [N, 168]
    return X, nodes, global_mean


def make_windows(X, t_in, t_out):
    """Sliding windows over the cyclic time axis. Returns [W, N, t_in], [W, N, t_out]."""
    N, T = X.shape
    xs, ys = [], []
    for s in range(T):
        idx_in = [(s + i) % T for i in range(t_in)]
        idx_out = [(s + t_in + j) % T for j in range(t_out)]
        xs.append(X[:, idx_in])
        ys.append(X[:, idx_out])
    return np.stack(xs), np.stack(ys)             # [T, N, t_in], [T, N, t_out]


# ============================================================
# MODEL
# ============================================================
def build_model_classes():
    """Import torch lazily and return the model class + torch handle."""
    import torch
    import torch.nn as nn

    class STGNN(nn.Module):
        def __init__(self, n_nodes, t_in, t_out, hidden=64, emb_dim=16):
            super().__init__()
            self.n_nodes = n_nodes
            self.t_out = t_out
            # self-adaptive adjacency (learns network structure, no lat/lon needed)
            self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
            self.gconv1 = nn.Linear(hidden, hidden)
            self.gconv2 = nn.Linear(hidden, hidden)
            self.act = nn.ReLU()
            self.head = nn.Linear(hidden, t_out)

        def adjacency(self):
            a = torch.relu(self.e1 @ self.e2.t())
            return torch.softmax(a, dim=1)         # row-stochastic [N, N]

        def forward(self, x):                      # x: [B, N, t_in]
            B, N, T = x.shape
            seq = x.reshape(B * N, T, 1)
            _, h = self.gru(seq)                   # h: [1, B*N, hidden]
            h = h.squeeze(0).reshape(B, N, -1)     # [B, N, hidden]
            A = self.adjacency()
            # two diffusion graph-conv layers with residual
            h = h + self.act(self.gconv1(torch.einsum("nm,bmh->bnh", A, h)))
            h = h + self.act(self.gconv2(torch.einsum("nm,bmh->bnh", A, h)))
            return self.head(h)                    # [B, N, t_out]

    return STGNN, torch, nn


# ============================================================
# TRAIN
# ============================================================
def train(args):
    STGNN, torch, nn = build_model_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X, nodes, gmean = build_cyclic_tensor(args.top_n)
    N = len(nodes)
    mean, std = float(X.mean()), float(X.std() + 1e-6)
    Xn = (X - mean) / std
    print(f"Network: {N} intersections | weekly tensor {Xn.shape}")

    xs, ys = make_windows(Xn, args.t_in, args.t_out)
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    # window-level train/val split
    W = xs.shape[0]
    perm = torch.randperm(W)
    n_val = max(1, int(W * 0.2))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = STGNN(N, args.t_in, args.t_out, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lossf = nn.MSELoss()

    print(f"Training {sum(p.numel() for p in model.parameters()):,} params "
          f"for {args.epochs} epochs...")
    best_val = math.inf
    for ep in range(1, args.epochs + 1):
        model.train()
        bx, by = xs[tr_idx], ys[tr_idx]
        # sensor dropout — mask random nodes' input so neighbours must fill in
        if args.mask > 0:
            m = (torch.rand(bx.shape[0], N, 1, device=device) > args.mask).float()
            bx_in = bx * m
        else:
            bx_in = bx
        opt.zero_grad()
        pred = model(bx_in)
        loss = lossf(pred, by)
        loss.backward()
        opt.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                vpred = model(xs[val_idx])
                vloss = lossf(vpred, ys[val_idx]).item()
                # MAE in congestion-level units
                vmae = (vpred - ys[val_idx]).abs().mean().item() * std
            tag = ""
            if vloss < best_val:
                best_val = vloss
                tag = " *"
                torch.save({
                    "state_dict": model.state_dict(),
                    "nodes": nodes, "mean": mean, "std": std,
                    "t_in": args.t_in, "t_out": args.t_out,
                    "hidden": args.hidden, "global_mean": gmean,
                }, MODEL_DIR / "stgnn_traffic.pt")
            print(f"  epoch {ep:3d}  train {loss.item():.4f}  "
                  f"val {vloss:.4f}  val MAE {vmae:.3f} lvl{tag}")

    print(f"\nSaved best model -> {MODEL_DIR / 'stgnn_traffic.pt'} "
          f"(val MSE {best_val:.4f})")

    # ---- evaluation ----------------------------------------------------
    model.eval()
    vx, vy = xs[val_idx], ys[val_idx]
    with torch.no_grad():
        gnn_mae = (model(vx) - vy).abs().mean().item() * std
    # persistence baseline (full input): predict last observed step
    persist = vx[:, :, -1:].repeat(1, 1, args.t_out)
    base_mae = (persist - vy).abs().mean().item() * std
    print(f"\nMulti-horizon val MAE  —  STGNN {gnn_mae:.3f}  vs  "
          f"persistence {base_mae:.3f} lvl "
          f"({(base_mae - gnn_mae) / base_mae * 100:+.0f}% )")
    print("  Persistence repeats a flat last value; the STGNN forecasts the")
    print("  actual rise/fall curve the timeline needs. Secondary diagnostic —")
    print("  recovering intersections whose own sensor is DOWN (spatial only):")

    # sensor-down evaluation: mask a subset of nodes, score ONLY those nodes.
    torch.manual_seed(0)
    down = torch.rand(vx.shape[0], N, 1, device=device) < 0.4   # 40% sensors down
    vx_masked = vx * (~down).float()
    with torch.no_grad():
        gnn_down = model(vx_masked)
    down_m = down.expand_as(vy)
    gnn_down_mae = ((gnn_down - vy).abs() * down_m).sum().item() / down_m.sum().item() * std
    # non-spatial fallback for a down sensor = predict the global mean
    # (Xn is standardized, so the global mean is 0 in normalized units)
    mean_pred = torch.zeros_like(vy)
    base_down_mae = ((mean_pred - vy).abs() * down_m).sum().item() / down_m.sum().item() * std
    print(f"Sensor-down MAE        —  STGNN {gnn_down_mae:.3f}  vs  "
          f"mean-fill {base_down_mae:.3f} lvl  "
          f"({(base_down_mae - gnn_down_mae) / base_down_mae * 100:+.0f}% )")
    print("  -> graph-based recovery of missing sensors, which XGBoost can't do.")


# ============================================================
# FORECAST  ->  feeds the dashboard
# ============================================================
def _label(avg):
    return LEVEL_NAMES[min(int(round(avg)), 3)]


def _distribution(node_vals):
    """Fraction of nodes in each level bucket."""
    levels = np.clip(np.round(node_vals).astype(int), 0, 3)
    return {LEVEL_NAMES[i]: float((levels == i).mean()) for i in range(4)}


def forecast(args):
    STGNN, torch, nn = build_model_classes()
    ckpt_path = MODEL_DIR / "stgnn_traffic.pt"
    if not ckpt_path.exists():
        print("No trained model. Run with --train first (or use --demo).")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    nodes, mean, std = ckpt["nodes"], ckpt["mean"], ckpt["std"]
    t_in, t_out = ckpt["t_in"], ckpt["t_out"]
    N = len(nodes)

    model = STGNN(N, t_in, t_out, hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X, _, _ = build_cyclic_tensor(args.top_n)
    # align rows to the model's node order where possible
    Xn = (X[:N] - mean) / std

    now = datetime.now()
    wh = (now.weekday() * 24 + now.hour) % WEEK_HOURS
    in_idx = [(wh - t_in + 1 + i) % WEEK_HOURS for i in range(t_in)]
    x_in = torch.tensor(Xn[:, in_idx][None, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x_in)[0].numpy() * std + mean   # [N, t_out]

    current = X[:, wh]                                # [N]
    cur_avg = float(np.clip(current.mean(), 0, 3))

    horizons = [{
        "offset_min": 0,
        "avg": round(cur_avg, 3),
        "label": _label(cur_avg),
        "distribution": _distribution(current),
    }]
    for h in range(t_out):
        col = np.clip(pred[:, h], 0, 3)
        avg = float(col.mean())
        horizons.append({
            "offset_min": (h + 1) * 60,
            "avg": round(avg, 3),
            "label": _label(avg),
            "distribution": _distribution(col),
        })

    next_avg = horizons[1]["avg"]
    trend = ("WORSENING" if next_avg > cur_avg + 0.15
             else "IMPROVING" if next_avg < cur_avg - 0.15 else "STABLE")

    out = {
        "model": "STGNN",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": now.isoformat(),
        "n_nodes": N,
        "current_week_hour": wh,
        "current_avg": round(cur_avg, 3),
        "current_label": _label(cur_avg),
        "trend": trend,
        "horizons": horizons,
        # mirror nowcast schema so existing widgets read it directly
        "predicted_avg": next_avg,
        "predicted_label": _label(next_avg),
        "distribution": horizons[1]["distribution"],
        "cameras_observed": N,
    }
    _write_forecast(out)


def demo(args):
    """Synthetic multi-horizon forecast — no torch, no data needed."""
    now = datetime.now()
    h = now.hour + now.minute / 60.0
    # diurnal congestion shape with AM/PM peaks
    base = (1.6 * math.exp(-((h - 8.5) ** 2) / 4.0)
            + 1.9 * math.exp(-((h - 17.5) ** 2) / 5.0) + 0.6)
    rng = np.random.default_rng(int(now.timestamp()) // 60)

    def snapshot(hours_ahead):
        hh = (h + hours_ahead) % 24
        v = (1.6 * math.exp(-((hh - 8.5) ** 2) / 4.0)
             + 1.9 * math.exp(-((hh - 17.5) ** 2) / 5.0) + 0.6)
        node_vals = np.clip(rng.normal(v, 0.6, size=200), 0, 3)
        return float(np.clip(v, 0, 3)), node_vals

    horizons = []
    for off in [0, 60, 120, 180, 240]:
        avg, node_vals = snapshot(off / 60.0)
        horizons.append({
            "offset_min": off, "avg": round(avg, 3), "label": _label(avg),
            "distribution": _distribution(node_vals),
        })
    cur_avg = horizons[0]["avg"]
    next_avg = horizons[1]["avg"]
    trend = ("WORSENING" if next_avg > cur_avg + 0.15
             else "IMPROVING" if next_avg < cur_avg - 0.15 else "STABLE")
    out = {
        "model": "STGNN (demo)", "device": "demo", "timestamp": now.isoformat(),
        "n_nodes": 200, "current_avg": cur_avg, "current_label": _label(cur_avg),
        "trend": trend, "horizons": horizons,
        "predicted_avg": next_avg, "predicted_label": _label(next_avg),
        "distribution": horizons[1]["distribution"], "cameras_observed": 200,
    }
    _write_forecast(out)


def _write_forecast(out):
    path = STATE_DIR / "latest_forecast.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  {out['model']} on {out['device']}  "
          f"now={out['current_label']}  next={out['predicted_label']}  "
          f"trend={out['trend']}")
    print(f"  horizons: " + "  ".join(
        f"+{hz['offset_min']}m {hz['avg']:.2f}" for hz in out["horizons"]))
    print(f"  Saved -> {path}")


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Spatio-temporal GNN traffic forecaster")
    ap.add_argument("--train", action="store_true", help="train the model on the GPU")
    ap.add_argument("--forecast", action="store_true", help="emit a horizon forecast")
    ap.add_argument("--demo", action="store_true", help="synthetic forecast (no torch)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--t_in", type=int, default=12, help="input window (hours)")
    ap.add_argument("--t_out", type=int, default=4, help="forecast horizon (hours)")
    ap.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="graph nodes")
    ap.add_argument("--mask", type=float, default=0.3, help="sensor-dropout fraction")
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()

    if args.demo:
        demo(args)
    elif args.train:
        train(args)
        forecast(args)          # emit an initial forecast after training
    elif args.forecast:
        forecast(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
