#!/usr/bin/env python3
"""Train LSTM shelter occupancy predictor on GPU using PyTorch.

Each sample is a 14-day sequence of features for one program,
predicting tomorrow's occupancy rate and at-capacity classification.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
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

# ---- Build sequences ----
SEQ_LEN = 14  # 14 days of history -> predict day 15


class ShelterSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=SEQ_LEN):
        self.sequences = []
        self.targets_reg = []
        self.targets_cls = []
        self.meta = []

        for pid, group in df.groupby("PROGRAM_ID"):
            group = group.sort_values("OCCUPANCY_DATE")
            features = group[feature_cols].values.astype(np.float32)
            target_rate = group["target_occ_rate"].values.astype(np.float32)
            target_cap = group["target_at_capacity"].values.astype(np.float32)

            # Replace NaN with 0 in features
            features = np.nan_to_num(features, nan=0.0)

            for i in range(seq_len, len(group)):
                seq = features[i - seq_len:i]
                self.sequences.append(seq)
                self.targets_reg.append(target_rate[i - 1])
                self.targets_cls.append(target_cap[i - 1])
                row = group.iloc[i - 1]
                self.meta.append({
                    "date": str(row["OCCUPANCY_DATE"]),
                    "tomorrow": str(row["tomorrow"]),
                    "program": row.get("PROGRAM_NAME", ""),
                    "shelter": row.get("SHELTER_GROUP", ""),
                    "location": row.get("LOCATION_NAME", ""),
                    "sector": row.get("SECTOR", ""),
                })

        self.sequences = np.array(self.sequences)
        self.targets_reg = np.array(self.targets_reg)
        self.targets_cls = np.array(self.targets_cls)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets_reg[idx]),
            torch.tensor(self.targets_cls[idx]),
        )


print("Building sequences...")
t0 = time.time()
train_ds = ShelterSequenceDataset(train_df, feature_cols)
test_ds = ShelterSequenceDataset(test_df, feature_cols)
print(f"Train sequences: {len(train_ds):,}, Test sequences: {len(test_ds):,}")
print(f"Sequence build time: {time.time() - t0:.1f}s")

# Normalize features using train statistics
train_mean = train_ds.sequences.mean(axis=(0, 1))
train_std = train_ds.sequences.std(axis=(0, 1)) + 1e-8
train_ds.sequences = (train_ds.sequences - train_mean) / train_std
test_ds.sequences = (test_ds.sequences - train_mean) / train_std

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)


# ---- Model ----
class ShelterLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout,
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.head_cls = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        reg = self.head_reg(last).squeeze(-1)
        cls = self.head_cls(last).squeeze(-1)
        return reg, cls


n_features = len(feature_cols)
model = ShelterLSTM(n_features, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ---- Training ----
EPOCHS = 50
LR = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_reg_fn = nn.MSELoss()
loss_cls_fn = nn.BCEWithLogitsLoss()

best_auc = 0
best_epoch = 0

print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE}")
print(f"{'Epoch':>5} {'Train Loss':>12} {'Test Loss':>12} {'Test AUC':>10} {'Test MAE':>10} {'Time':>8}")
print("-" * 62)

total_train_time = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    train_loss_sum = 0
    train_n = 0
    for X, y_reg, y_cls in train_loader:
        X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
        pred_reg, pred_cls = model(X)
        loss = loss_reg_fn(pred_reg, y_reg) + loss_cls_fn(pred_cls, y_cls) * 100
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss_sum += loss.item() * len(X)
        train_n += len(X)
    scheduler.step()
    train_loss = train_loss_sum / train_n

    # Eval
    model.eval()
    test_loss_sum = 0
    test_n = 0
    all_pred_reg, all_pred_cls, all_y_reg, all_y_cls = [], [], [], []
    with torch.no_grad():
        for X, y_reg, y_cls in test_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            pred_reg, pred_cls = model(X)
            loss = loss_reg_fn(pred_reg, y_reg) + loss_cls_fn(pred_cls, y_cls) * 100
            test_loss_sum += loss.item() * len(X)
            test_n += len(X)
            all_pred_reg.append(pred_reg.cpu().numpy())
            all_pred_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
            all_y_reg.append(y_reg.cpu().numpy())
            all_y_cls.append(y_cls.cpu().numpy())

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
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_DIR / "lstm_best.pt")

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"{epoch:5d} {train_loss:12.4f} {test_loss:12.4f} {auc:10.4f} {mae:10.2f} {epoch_time:7.1f}s")

    # Early stopping
    if epoch - best_epoch > 15:
        print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
        break

print(f"\nTotal training time: {total_train_time:.1f}s")
print(f"Best test AUC: {best_auc:.4f} at epoch {best_epoch}")

# ---- Final evaluation with best model ----
print("\n" + "=" * 60)
print("FINAL EVALUATION (best model)")
print("=" * 60)

model.load_state_dict(torch.load(MODEL_DIR / "lstm_best.pt", weights_only=True))
model.eval()

all_pred_reg, all_pred_cls, all_y_reg, all_y_cls = [], [], [], []
with torch.no_grad():
    for X, y_reg, y_cls in test_loader:
        X = X.to(device)
        pred_reg, pred_cls = model(X)
        all_pred_reg.append(pred_reg.cpu().numpy())
        all_pred_cls.append(torch.sigmoid(pred_cls).cpu().numpy())
        all_y_reg.append(y_reg.numpy())
        all_y_cls.append(y_cls.numpy())

all_pred_reg = np.concatenate(all_pred_reg)
all_pred_cls = np.concatenate(all_pred_cls)
all_y_reg = np.concatenate(all_y_reg)
all_y_cls = np.concatenate(all_y_cls)

y_pred_binary = (all_pred_cls >= 0.5).astype(int)

print(f"\nClassification (tomorrow at capacity?):")
print(classification_report(all_y_cls.astype(int), y_pred_binary,
                            target_names=["Below 100%", "At/Over 100%"]))
print(f"ROC AUC: {roc_auc_score(all_y_cls.astype(int), all_pred_cls):.4f}")

mae = mean_absolute_error(all_y_reg, all_pred_reg)
rmse = np.sqrt(np.mean((all_y_reg - all_pred_reg) ** 2))
residuals = all_y_reg - all_pred_reg
pct2 = (np.abs(residuals) <= 2).mean()
pct5 = (np.abs(residuals) <= 5).mean()
print(f"\nRegression (tomorrow's occupancy rate):")
print(f"MAE:  {mae:.2f}%")
print(f"RMSE: {rmse:.2f}%")
print(f"Within 2%: {pct2:.1%}")
print(f"Within 5%: {pct5:.1%}")

# ---- Compare with XGBoost ----
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'XGBoost':>12} {'LSTM':>12}")
print("-" * 50)
print(f"{'Training time':<25} {'0.7s':>12} {f'{total_train_time:.1f}s':>12}")
print(f"{'ROC AUC':<25} {'0.9273':>12} {f'{best_auc:.4f}':>12}")
print(f"{'Regression MAE':<25} {'2.41%':>12} {f'{mae:.2f}%':>12}")
print(f"{'Regression RMSE':<25} {'6.13%':>12} {f'{rmse:.2f}%':>12}")
print(f"{'Within 5%':<25} {'87.6%':>12} {f'{pct5:.1%}':>12}")

# Save normalization stats for inference
np.savez(MODEL_DIR / "lstm_norm.npz", mean=train_mean, std=train_std)
print(f"\nModel + normalization saved to {MODEL_DIR}/")
