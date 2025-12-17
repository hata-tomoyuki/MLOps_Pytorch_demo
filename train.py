import argparse
import glob
import os
import shutil  # SageMakerのmodel_dirへinference.pyをコピーするために使用

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

# ============================================================
# 再現性（乱数シード）設定
# - numpy / torch / lightning で同じ seed を使うことで、学習結果のブレを抑える
# - SageMaker で複数回回したときの比較がしやすい
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
pl.seed_everything(RANDOM_STATE, workers=True)

# ============================================================
# データ定義
# - target_col: 目的変数（回帰なので連続値）
# - feature_cols: モデルに入力する特徴量
# ============================================================
target_col = "sprint_time"

feature_cols = [
    "sex", "age", "height", "weight", "reaction_time", "lane",
    "temperature", "humidity", "wind_speed", "wind_direction",
    "altitude", "track_condition", "season",
]

# 数値特徴量：標準化(StandardScaler)する
numeric_features = [
    "age", "height", "weight", "reaction_time", "lane",
    "temperature", "humidity", "wind_speed", "altitude",
]

# カテゴリ特徴量：OneHotEncoder する（未知カテゴリは無視）
categorical_features = [
    "sex", "wind_direction", "track_condition", "season",
]


def create_preprocess_pipeline():
    """
    前処理パイプラインを作成（学習時と推論時で同じ処理を保証するため保存する）
    - 数値: StandardScaler
    - カテゴリ: OneHotEncoder(handle_unknown='ignore')
    """
    # 数値列の変換（平均0・分散1に正規化）
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    # カテゴリ列の変換（one-hot）
    # sparse_output=False: numpy配列で返す（後段の torch に流しやすい）
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    # 列ごとに異なる変換を適用する
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess


class NumpyDataset(Dataset):
    """
    numpy配列（またはpandas由来）を PyTorch Dataset にする薄いラッパー
    - DataLoader でミニバッチ学習できるようにする
    """

    def __init__(self, X, y):
        # Xは (N, D) を想定。float32 に揃える（GPU/CPUで扱いやすい）
        self.X = np.asarray(X, dtype=np.float32)
        # yは (N,) を想定。float32 に揃える
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        # データ数
        return len(self.X)

    def __getitem__(self, idx):
        # PyTorch Tensor に変換して返す
        # yは scalar を想定し tensor化
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TabularLightningRegressor(pl.LightningModule):
    """
    表形式データ向けのシンプルなMLP回帰モデル（PyTorch Lightning）
    - forward: 推論
    - training_step / validation_step: 損失計算＆ログ
    - configure_optimizers: 最適化手法の定義
    """

    def __init__(self, n_features, hidden_dim=64, lr=1e-3):
        super().__init__()
        # Lightningが hparams を自動保存できるようにする（ログ用途）
        self.save_hyperparameters()

        # 多層パーセプトロン（MLP）
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 回帰なので出力は1次元
        )

        # 回帰の代表的損失（MSE）
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # (N, 1) -> (N,) にするため squeeze(-1)
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # 1ステップ分の学習処理
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        # prog_bar=True: プログレスバーに表示
        # on_epoch=True: epoch単位の集計も出す
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 検証データでの損失
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 最適化手法（Adam）
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_one_run(
    X_train_np, y_train_np,
    X_valid_np, y_valid_np,
    hidden_dim=64, lr=1e-3,
    max_epochs=50, batch_size=64,
):
    """
    1回分の学習を実行して LightningModule を返す
    - 前処理済みの numpy 配列を受け取る想定
    """
    # 念のためここでもseed（学習関数単体で呼ばれても再現性を確保）
    pl.seed_everything(RANDOM_STATE, workers=True)

    # DataLoader を作成（ミニバッチ化）
    train_loader = DataLoader(
        NumpyDataset(X_train_np, y_train_np),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NumpyDataset(X_valid_np, y_valid_np),
        batch_size=batch_size, shuffle=False
    )

    # モデル作成（入力次元は前処理後の列数）
    model = TabularLightningRegressor(
        n_features=X_train_np.shape[1],
        hidden_dim=hidden_dim,
        lr=lr,
    )

    # Lightning Trainer
    # - accelerator="auto": CPU/GPUを自動選択
    # - devices=1: 単一デバイスで学習（分散学習しない想定）
    # - logger=False: ここでは外部ロガーを使わない
    # - enable_checkpointing=False: ckptをLightningに任せず、後段で torch.save する
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    # 学習実行
    trainer.fit(model, train_loader, val_loader)

    return model


def main(args):
    """
    SageMaker Training Job の entry point を想定したメイン処理

    SageMakerの慣習:
    - 学習データ: SM_CHANNEL_TRAIN (/opt/ml/input/data/train) に配置される
    - モデル出力: SM_MODEL_DIR (/opt/ml/model) に保存する
    """

    # ============================================================
    # 1) CSV読み込み
    # - SageMaker では channel=train のパスが環境変数に入る
    # - ローカル実行のときはデフォルトを /opt/ml/input/data/train にしておく
    # ============================================================
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    # train channel 配下の CSV を探す（最初の1件を利用）
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV found in {input_dir}")
    csv_path = csv_files[0]

    df = pd.read_csv(csv_path)

    # 特徴量と目的変数を切り出し
    X = df[feature_cols]
    y = df[target_col]

    # ============================================================
    # 2) 学習/検証に分割
    # ============================================================
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ============================================================
    # 3) 前処理（学習データでfitし、検証データはtransformのみ）
    # - ここ重要: valid で fit しない（データリーク防止）
    # ============================================================
    preprocess = create_preprocess_pipeline()
    X_tr_proc = preprocess.fit_transform(X_train)
    X_val_proc = preprocess.transform(X_valid)

    # ============================================================
    # 4) 学習
    # ============================================================
    model = train_one_run(
        X_tr_proc, y_train.values,
        X_val_proc, y_valid.values,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    # ============================================================
    # 5) モデル保存（SageMaker規約: /opt/ml/model 以下に保存）
    # - Serving時に必要な情報を一式残す
    #   ① model.pt（重み + 入力次元等）
    #   ② preprocess.pkl（前処理）
    #   ③ code/inference.py（推論エントリーポイント; 任意）
    # ============================================================
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    # 推論用に eval モードへ
    model.eval()

    # LightningModule の中の nn.Sequential（純粋な PyTorch モデル部）だけ取り出す
    # ※ 推論側で Lightning を使わない設計にしている前提
    core_model = model.model.cpu()

    # 推論時に必要になるメタ情報（入力次元・隠れ層次元など）を明示的に保存
    n_features = X_tr_proc.shape[1]
    hidden_dim = args.hidden_dim

    ckpt = {
        "model_state_dict": core_model.state_dict(),
        "n_features": n_features,
        "hidden_dim": hidden_dim,
    }

    # ① PyTorch重み保存（推論側でtorch.loadして復元する想定）
    torch.save(ckpt, os.path.join(model_dir, "model.pt"))

    # ② 前処理（sklearn）も保存（推論時に同じ変換を再現するため）
    joblib.dump(preprocess, os.path.join(model_dir, "preprocess.pkl"))

    # ============================================================
    # ③ inference.py を model_dir/code に配置（あれば）
    # - SageMakerのPyTorch推論コンテナは /opt/ml/model/code/ に置かれた
    #   inference.py を読み込む構成が一般的（source_dir側に同梱するのも可）
    # - ここでは「プロジェクト内に存在すればコピーする」実装
    # ============================================================
    code_dir = os.path.join(model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # train.py 自身の配置ディレクトリ（SageMakerの source_dir 展開場所）
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # inference.py を探す候補（プロジェクト構成に合わせて増やせる）
    candidate_paths = [
        os.path.join(current_dir, "inference.py"),                  # train.py と同じ階層
        os.path.join(current_dir, "code", "inference.py"),          # ./code/inference.py
        os.path.join(current_dir, "model", "code", "inference.py"), # ./model/code/inference.py
    ]

    src_inference = None
    for p in candidate_paths:
        if os.path.exists(p):
            src_inference = p
            break

    if src_inference is None:
        # inference.py が無くても学習は成功させる（推論はデフォルト挙動 or 別途指定）
        print("[WARN] inference.py not found. Searched paths:")
        for p in candidate_paths:
            print(f"  - {p}")
        print(
            "[WARN] Skipping copy of inference.py. "
            "If you want custom inference, include inference.py in source_dir."
        )
    else:
        # 見つかった inference.py を model_dir/code へコピー
        dst_inference = os.path.join(code_dir, "inference.py")
        shutil.copyfile(src_inference, dst_inference)
        print(f"Copied inference.py from {src_inference} to {dst_inference}")

    print("Saved model.pt (with n_features, hidden_dim), preprocess.pkl and maybe code/inference.py")


if __name__ == "__main__":
    # ============================================================
    # SageMaker Training Job から渡されるハイパーパラメータを CLI で受け取る
    # 例:
    #   python train.py --batch_size 64 --hidden_dim 64 --lr 0.001 --max_epochs 50
    # ============================================================
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden layer size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")

    args = parser.parse_args()
    main(args)
