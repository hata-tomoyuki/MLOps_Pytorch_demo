import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
pl.seed_everything(RANDOM_STATE, workers=True)

# ======== データの定義 ========
target_col = "sprint_time"

feature_cols = [
    "sex", "age", "height", "weight", "reaction_time", "lane",
    "temperature", "humidity", "wind_speed", "wind_direction",
    "altitude", "track_condition", "season",
]

numeric_features = [
    "age", "height", "weight", "reaction_time", "lane",
    "temperature", "humidity", "wind_speed", "altitude",
]

categorical_features = [
    "sex", "wind_direction", "track_condition", "season",
]


def create_preprocess_pipeline():
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TabularLightningRegressor(pl.LightningModule):
    def __init__(self, n_features, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_one_run(
    X_train_np, y_train_np,
    X_valid_np, y_valid_np,
    hidden_dim=64, lr=1e-3,
    max_epochs=50, batch_size=64,
):
    pl.seed_everything(RANDOM_STATE, workers=True)

    train_loader = DataLoader(
        NumpyDataset(X_train_np, y_train_np),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NumpyDataset(X_valid_np, y_valid_np),
        batch_size=batch_size, shuffle=False
    )

    model = TabularLightningRegressor(
        n_features=X_train_np.shape[1],
        hidden_dim=hidden_dim,
        lr=lr,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return model


def main(args):

    # ===== Load CSV =====
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV found in {input_dir}")
    csv_path = csv_files[0]

    df = pd.read_csv(csv_path)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ==== preprocess ====
    preprocess = create_preprocess_pipeline()
    X_tr_proc = preprocess.fit_transform(X_train)
    X_val_proc = preprocess.transform(X_valid)

    # ==== training ====
    model = train_one_run(
        X_tr_proc, y_train.values,
        X_val_proc, y_valid.values,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    # ==== SAVE TO SM_MODEL_DIR ====
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    model.eval()
    core_model = model.model.cpu()  # nn.Sequential 部分

    # ハイパーパラメータを明示的に保存
    n_features = X_tr_proc.shape[1]
    hidden_dim = args.hidden_dim

    ckpt = {
        "model_state_dict": core_model.state_dict(),
        "n_features": n_features,
        "hidden_dim": hidden_dim,
    }

    torch.save(ckpt, os.path.join(model_dir, "model.bin"))

    # 前処理も保存
    joblib.dump(preprocess, os.path.join(model_dir, "preprocess.pkl"))

    print("Saved model.pt (with n_features, hidden_dim) and preprocess.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
