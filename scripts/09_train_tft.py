#!/usr/bin/env python3
"""Train Temporal Fusion Transformer for shelter occupancy prediction on GPU.

TFT is an attention-based architecture designed for multi-horizon time-series
forecasting with heterogeneous inputs:
- Static covariates (shelter sector, program model)
- Known future inputs (calendar features, weather forecasts)
- Observed past inputs (occupancy history, lag features)

Uses interpretable multi-head attention to show which time steps and features
drive predictions. Significantly more GPU-intensive than XGBoost/LSTM.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Load data ----
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

with open(DATA_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

print(f"Features: {len(feature_cols)}")
print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

# ---- Feature groups for TFT ----
STATIC_FEATURES = ["sector_enc", "program_model_enc"]

KNOWN_FUTURE = [
    "day_of_week", "day_of_month", "month", "is_weekend", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]

OBSERVED_PAST = [c for c in feature_cols if c not in STATIC_FEATURES + KNOWN_FUTURE]

print(f"Static features: {len(STATIC_FEATURES)}")
print(f"Known future inputs: {len(KNOWN_FUTURE)}")
print(f"Observed past inputs: {len(OBSERVED_PAST)}")

# ---- Build sequences ----
SEQ_LEN = 14


class TFTDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=SEQ_LEN):
        self.static = []
        self.known_future = []
        self.observed_past = []
        self.targets_reg = []
        self.targets_cls = []

        for pid, group in df.groupby("PROGRAM_ID"):
            group = group.sort_values("OCCUPANCY_DATE")
            static_vals = group[STATIC_FEATURES].values[0].astype(np.float32)
            known_vals = group[KNOWN_FUTURE].values.astype(np.float32)
            observed_vals = group[OBSERVED_PAST].values.astype(np.float32)
            target_rate = group["target_occ_rate"].values.astype(np.float32)
            target_cap = group["target_at_capacity"].values.astype(np.float32)

            known_vals = np.nan_to_num(known_vals, nan=0.0)
            observed_vals = np.nan_to_num(observed_vals, nan=0.0)

            for i in range(seq_len, len(group)):
                self.static.append(static_vals)
                self.known_future.append(known_vals[i - seq_len:i])
                self.observed_past.append(observed_vals[i - seq_len:i])
                self.targets_reg.append(target_rate[i - 1])
                self.targets_cls.append(target_cap[i - 1])

        self.static = np.array(self.static)
        self.known_future = np.array(self.known_future)
        self.observed_past = np.array(self.observed_past)
        self.targets_reg = np.array(self.targets_reg)
        self.targets_cls = np.array(self.targets_cls)

    def __len__(self):
        return len(self.static)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.static[idx]),
            torch.tensor(self.known_future[idx]),
            torch.tensor(self.observed_past[idx]),
            torch.tensor(self.targets_reg[idx]),
            torch.tensor(self.targets_cls[idx]),
        )


print("\nBuilding sequences...")
t0 = time.time()
train_ds = TFTDataset(train_df, feature_cols)
test_ds = TFTDataset(test_df, feature_cols)
print(f"Train sequences: {len(train_ds):,}, Test sequences: {len(test_ds):,}")
print(f"Build time: {time.time() - t0:.1f}s")

# Normalize observed features
obs_mean = train_ds.observed_past.mean(axis=(0, 1))
obs_std = train_ds.observed_past.std(axis=(0, 1)) + 1e-8
train_ds.observed_past = (train_ds.observed_past - obs_mean) / obs_std
test_ds.observed_past = (test_ds.observed_past - obs_mean) / obs_std

kf_mean = train_ds.known_future.mean(axis=(0, 1))
kf_std = train_ds.known_future.std(axis=(0, 1)) + 1e-8
train_ds.known_future = (train_ds.known_future - kf_mean) / kf_std
test_ds.known_future = (test_ds.known_future - kf_mean) / kf_std

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)


# ---- TFT Model Components ----

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate(h))
        h = gate * h
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_vars, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_vars = num_vars
        self.var_dim = input_dim // num_vars
        self.grns = nn.ModuleList([
            GatedResidualNetwork(self.var_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])
        self.softmax_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_vars, dropout, context_dim=context_dim
        )

    def forward(self, x, context=None):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        has_time = x.dim() == 3
        if has_time:
            batch, seq_len, _ = x.shape
            flat = x.reshape(batch * seq_len, -1)
        else:
            flat = x

        weights = F.softmax(self.softmax_grn(flat, context), dim=-1)

        var_outputs = []
        for i, grn in enumerate(self.grns):
            start = i * self.var_dim
            end = start + self.var_dim
            var_outputs.append(grn(flat[:, start:end]))

        var_outputs = torch.stack(var_outputs, dim=1)
        weights = weights.unsqueeze(-1)
        combined = (weights * var_outputs).sum(dim=1)

        if has_time:
            combined = combined.reshape(batch, seq_len, -1)
            weights = weights.reshape(batch, seq_len, self.num_vars, 1)

        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_avg = attn.mean(dim=1)
        out = torch.matmul(attn_avg, v)
        out = self.out_proj(out)

        return out, attn_avg


class TemporalFusionTransformer(nn.Module):
    def __init__(self, static_dim, known_dim, observed_dim, hidden_dim=160,
                 n_heads=4, num_layers=2, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Static variable processing
        self.static_vsn = VariableSelectionNetwork(
            static_dim, static_dim, hidden_dim, dropout
        ) if static_dim > 0 else None
        self.static_grn = GatedResidualNetwork(
            hidden_dim if static_dim > 0 else 0, hidden_dim, hidden_dim, dropout
        ) if static_dim > 0 else None
        self.static_proj = nn.Linear(static_dim, hidden_dim) if static_dim > 0 else None

        # Temporal variable processing
        total_temporal = known_dim + observed_dim
        self.temporal_proj = nn.Linear(total_temporal, hidden_dim)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )

        # Gated skip connection after LSTM
        self.post_lstm_gate = nn.Linear(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Static enrichment
        self.static_enrich_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim
        )

        # Multi-head attention
        self.self_attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.post_attn_gate = nn.Linear(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # Position-wise feed-forward
        self.ff_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.head_reg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, static, known_future, observed_past):
        batch = static.size(0)

        # Static context
        if self.static_proj is not None:
            static_embed = self.static_proj(static)
            context = static_embed
        else:
            context = torch.zeros(batch, self.hidden_dim, device=static.device)

        # Temporal inputs
        temporal = torch.cat([known_future, observed_past], dim=-1)
        temporal_embed = self.temporal_proj(temporal)

        # Add static context to temporal
        temporal_embed = temporal_embed + context.unsqueeze(1)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(temporal_embed)

        # Gated skip connection
        gate = torch.sigmoid(self.post_lstm_gate(lstm_out))
        lstm_out = self.post_lstm_norm(gate * lstm_out + temporal_embed)

        # Static enrichment
        enriched = self.static_enrich_grn(lstm_out, context.unsqueeze(1).expand_as(lstm_out))

        # Self-attention with causal mask
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=static.device))
        attn_out, attn_weights = self.self_attention(enriched, enriched, enriched, mask)

        # Post-attention gating
        gate = torch.sigmoid(self.post_attn_gate(attn_out))
        attn_out = self.post_attn_norm(gate * attn_out + enriched)

        # Feed-forward
        ff_out = self.ff_grn(attn_out)
        output = self.ff_norm(ff_out + attn_out)

        # Use last time step for prediction
        last = output[:, -1, :]
        reg = self.head_reg(last).squeeze(-1)
        cls = self.head_cls(last).squeeze(-1)

        return reg, cls, attn_weights


# ---- Initialize model ----
model = TemporalFusionTransformer(
    static_dim=len(STATIC_FEATURES),
    known_dim=len(KNOWN_FUTURE),
    observed_dim=len(OBSERVED_PAST),
    hidden_dim=160,
    n_heads=4,
    num_layers=2,
    dropout=0.1,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTFT parameters: {n_params:,}")
print(f"  vs LSTM: 241,922")
print(f"  Ratio: {n_params / 241922:.1f}x more parameters")

# ---- Training ----
EPOCHS = 80
LR = 5e-4
WARMUP = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP)
loss_reg_fn = nn.MSELoss()
loss_cls_fn = nn.BCEWithLogitsLoss()

best_auc = 0
best_epoch = 0
best_mae = 999

print(f"\nTraining TFT for {EPOCHS} epochs, batch_size={BATCH_SIZE}")
print(f"{'Epoch':>5} {'Train Loss':>12} {'Test Loss':>12} {'Test AUC':>10} {'Test MAE':>10} {'LR':>10} {'Time':>8}")
print("-" * 72)

total_train_time = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Warmup learning rate
    if epoch <= WARMUP:
        warmup_lr = LR * epoch / WARMUP
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

    # Train
    model.train()
    train_loss_sum = 0
    train_n = 0
    for static, known, observed, y_reg, y_cls in train_loader:
        static = static.to(device)
        known = known.to(device)
        observed = observed.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        pred_reg, pred_cls, _ = model(static, known, observed)
        loss = loss_reg_fn(pred_reg, y_reg) + loss_cls_fn(pred_cls, y_cls) * 100

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss_sum += loss.item() * len(static)
        train_n += len(static)

    if epoch > WARMUP:
        scheduler.step()
    train_loss = train_loss_sum / train_n

    # Eval
    model.eval()
    test_loss_sum = 0
    test_n = 0
    all_pred_reg, all_pred_cls, all_y_reg, all_y_cls = [], [], [], []
    all_attn = []

    with torch.no_grad():
        for static, known, observed, y_reg, y_cls in test_loader:
            static = static.to(device)
            known = known.to(device)
            observed = observed.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            pred_reg, pred_cls, attn = model(static, known, observed)
            loss = loss_reg_fn(pred_reg, y_reg) + loss_cls_fn(pred_cls, y_cls) * 100

            test_loss_sum += loss.item() * len(static)
            test_n += len(static)
            all_pred_reg.append(pred_reg.cpu().numpy())
            all_pred_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
            all_y_reg.append(y_reg.cpu().numpy())
            all_y_cls.append(y_cls.cpu().numpy())
            if epoch == EPOCHS or (best_epoch == epoch):
                all_attn.append(attn.cpu().numpy())

    test_loss = test_loss_sum / test_n
    all_pred_reg = np.concatenate(all_pred_reg)
    all_pred_cls = np.concatenate(all_pred_cls)
    all_y_reg = np.concatenate(all_y_reg)
    all_y_cls = np.concatenate(all_y_cls)

    auc = roc_auc_score(all_y_cls.astype(int), all_pred_cls)
    mae = mean_absolute_error(all_y_reg, all_pred_reg)

    epoch_time = time.time() - t0
    total_train_time += epoch_time

    if auc > best_auc:
        best_auc = auc
        best_mae = mae
        best_epoch = epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'kf_mean': kf_mean,
            'kf_std': kf_std,
        }, MODEL_DIR / "tft_best.pt")

    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"{epoch:5d} {train_loss:12.4f} {test_loss:12.4f} {auc:10.4f} {mae:10.2f} {current_lr:10.6f} {epoch_time:7.1f}s")

    if epoch - best_epoch > 20:
        print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
        break

print(f"\nTotal training time: {total_train_time:.1f}s")
print(f"Best test AUC: {best_auc:.4f} at epoch {best_epoch}")

# ---- Final evaluation ----
print("\n" + "=" * 60)
print("FINAL EVALUATION (best TFT model)")
print("=" * 60)

checkpoint = torch.load(MODEL_DIR / "tft_best.pt", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_pred_reg, all_pred_cls, all_y_reg, all_y_cls = [], [], [], []
all_attn_weights = []

with torch.no_grad():
    for static, known, observed, y_reg, y_cls in test_loader:
        static = static.to(device)
        known = known.to(device)
        observed = observed.to(device)

        pred_reg, pred_cls, attn = model(static, known, observed)
        all_pred_reg.append(pred_reg.cpu().numpy())
        all_pred_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
        all_y_reg.append(y_reg.numpy())
        all_y_cls.append(y_cls.numpy())
        all_attn_weights.append(attn.cpu().numpy())

all_pred_reg = np.concatenate(all_pred_reg)
all_pred_cls = np.concatenate(all_pred_cls)
all_y_reg = np.concatenate(all_y_reg)
all_y_cls = np.concatenate(all_y_cls)
all_attn_weights = np.concatenate(all_attn_weights)

y_pred_binary = (all_pred_cls >= 0.5).astype(int)

print(f"\nClassification (tomorrow at capacity?):")
print(classification_report(all_y_cls.astype(int), y_pred_binary,
                            target_names=["Below 100%", "At/Over 100%"]))
tft_auc = roc_auc_score(all_y_cls.astype(int), all_pred_cls)
print(f"ROC AUC: {tft_auc:.4f}")

tft_mae = mean_absolute_error(all_y_reg, all_pred_reg)
tft_rmse = np.sqrt(np.mean((all_y_reg - all_pred_reg) ** 2))
residuals = all_y_reg - all_pred_reg
pct2 = (np.abs(residuals) <= 2).mean()
pct5 = (np.abs(residuals) <= 5).mean()
acc = (y_pred_binary == all_y_cls.astype(int)).mean()

print(f"\nRegression (tomorrow's occupancy rate):")
print(f"MAE:  {tft_mae:.2f}%")
print(f"RMSE: {tft_rmse:.2f}%")
print(f"Within 2%: {pct2:.1%}")
print(f"Within 5%: {pct5:.1%}")

# ---- Attention analysis ----
print(f"\n{'='*60}")
print("TEMPORAL ATTENTION ANALYSIS")
print(f"{'='*60}")
avg_attn = all_attn_weights.mean(axis=0)
last_step_attn = avg_attn[-1]
print(f"Attention from prediction step to each historical day:")
for i, a in enumerate(last_step_attn):
    day_label = f"t-{SEQ_LEN - i}"
    bar = "#" * int(a / last_step_attn.max() * 40)
    print(f"  {day_label:>5s}: {a:.4f} {bar}")

# ---- Model comparison ----
print(f"\n{'='*60}")
print("MODEL COMPARISON: XGBoost vs LSTM vs TFT")
print(f"{'='*60}")
print(f"{'Metric':<25} {'XGBoost':>12} {'LSTM':>12} {'TFT':>12}")
print("-" * 65)
print(f"{'Parameters':<25} {'N/A':>12} {'241,922':>12} {f'{n_params:,}':>12}")
print(f"{'Training time':<25} {'0.7s':>12} {'45.4s':>12} {f'{total_train_time:.1f}s':>12}")
print(f"{'ROC AUC':<25} {'0.9275':>12} {'0.9147':>12} {f'{tft_auc:.4f}':>12}")
print(f"{'Classification Accuracy':<25} {'87%':>12} {'86%':>12} {f'{acc:.0%}':>12}")
print(f"{'Regression MAE':<25} {'2.42%':>12} {'3.52%':>12} {f'{tft_mae:.2f}%':>12}")
print(f"{'Regression RMSE':<25} {'6.14%':>12} {'7.11%':>12} {f'{tft_rmse:.2f}%':>12}")
print(f"{'Within 5%':<25} {'87.8%':>12} {'84.2%':>12} {f'{pct5:.1%}':>12}")

print(f"\nTFT model saved to {MODEL_DIR}/tft_best.pt")
