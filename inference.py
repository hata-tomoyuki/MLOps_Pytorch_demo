import json
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# OneHotEncoder の categories_ に混入した NaN / None を除去するパッチ
# ============================================================
# 背景:
# - 学習時のデータや pandas / sklearn のバージョン差異により
#   OneHotEncoder.categories_ に NaN / None / pd.NA が残ることがある
# - 推論時に transform() でエラーになるケースを防ぐため、
#   「欠損値だけ」を安全に除去する
#
# ポイント:
# - 文字列カテゴリ（"M", "F" など）は削除しない
# - sklearn の内部構造（Pipeline / ColumnTransformer）を再帰的に走査
# ============================================================
def _patch_onehot_categories_no_nan(preprocess: Any) -> None:
    """
    OneHotEncoder.categories_ から、欠損値（NaN / None / pandas.NA）のみを安全に除去する。
    文字列カテゴリなどは除去しない。
    """
    import pandas as pd

    def is_invalid(v):
        # None / pandas.NA
        if v is None or v is pd.NA:
            return True
        # float の NaN
        if isinstance(v, float) and np.isnan(v):
            return True
        # それ以外（文字列・数値など）は有効カテゴリ
        return False

    def _patch_encoder(enc: OneHotEncoder) -> None:
        if not hasattr(enc, "categories_"):
            return
        new_cats = []
        for cats in enc.categories_:
            filtered = []
            for c in cats:
                if not is_invalid(c):
                    filtered.append(c)
            new_cats.append(np.array(filtered, dtype=object))
        enc.categories_ = new_cats

    def _walk(obj: Any) -> None:
        # OneHotEncoder 単体
        if isinstance(obj, OneHotEncoder):
            _patch_encoder(obj)
        # Pipeline 内を再帰的に走査
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                _walk(step)
        # ColumnTransformer 内の transformer を走査
        if isinstance(obj, ColumnTransformer):
            transformers_list = getattr(obj, "transformers_", None) or getattr(obj, "transformers", [])
            for _, transformer, _ in transformers_list:
                _walk(transformer)

    _walk(preprocess)


# ============================================================
# 学習時と完全に同じ feature の順序を定義
# ============================================================
# ※ 推論時にカラム順がずれると、モデル入力が破綻するため超重要
FEATURE_COLUMNS: List[str] = [
    "sex",
    "age",
    "height",
    "weight",
    "reaction_time",
    "lane",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "altitude",
    "track_condition",
    "season",
]


# ============================================================
# SageMaker 推論コンテナ必須フック
#   - model_fn
#   - input_fn
#   - predict_fn
#   - output_fn
# ============================================================

def model_fn(model_dir: str):
    """
    コンテナ起動時に 1 回だけ呼ばれる初期化関数。

    役割:
    - /opt/ml/model 以下から
      - 学習済みモデル（model.pt）
      - 前処理（preprocess.pkl）
      を読み込む
    - 推論に必要なオブジェクトをメモリに常駐させる
    """
    print(f"[model_fn] model_dir = {model_dir}")
    files = os.listdir(model_dir)
    print(f"[model_fn] files in model_dir: {files}")

    model_path = os.path.join(model_dir, "model.pt")
    preprocess_path = os.path.join(model_dir, "preprocess.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(f"model.pt not found in {model_dir}")
    if not os.path.exists(preprocess_path):
        raise RuntimeError(f"preprocess.pkl not found in {model_dir}")

    # ===== 前処理パイプラインのロード =====
    preprocess = joblib.load(preprocess_path)
    print("[model_fn] preprocess.pkl loaded")

    # OneHotEncoder.categories_ から欠損値だけ除去（安全対策）
    _patch_onehot_categories_no_nan(preprocess)
    print("[model_fn] preprocess patched (removed NaN/None from OneHotEncoder.categories_)")

    # ===== モデル checkpoint のロード =====
    ckpt = torch.load(model_path, map_location="cpu")

    # 新形式（train.py で保存している推奨形式）
    # {
    #   "model_state_dict": ...,
    #   "n_features": int,
    #   "hidden_dim": int
    # }
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        n_features = int(ckpt["n_features"])
        hidden_dim = int(ckpt.get("hidden_dim", 64))
        print(f"[model_fn] loaded from new-format ckpt: n_features={n_features}, hidden_dim={hidden_dim}")
    else:
        # ===== 後方互換対応 =====
        # - 昔の実験で state_dict だけ保存していた場合
        # - Lightning の checkpoint 形式 {"state_dict": {...}} の場合
        state_dict = ckpt

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if "0.weight" in state_dict:
            # nn.Sequential の state_dict をそのまま保存していたケース
            first_linear_weight = state_dict["0.weight"]
        elif "model.0.weight" in state_dict:
            # LightningModule 全体を保存していたケース
            first_linear_weight = state_dict["model.0.weight"]

            # "model." プレフィックスを削除して nn.Sequential に合わせる
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        else:
            raise RuntimeError(
                f"Unexpected state_dict keys example: {list(state_dict.keys())[:10]}"
            )

        # 入力次元は weight の shape から推定
        n_features = first_linear_weight.shape[1]
        hidden_dim = 64
        print(f"[model_fn] backward-compat: inferred n_features={n_features}, hidden_dim={hidden_dim}")

    # ===== nn.Sequential モデルを再構築 =====
    model = nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("[model_fn] model loaded successfully")

    # predict_fn に渡すため dict で返す
    return {"model": model, "preprocess": preprocess}


def input_fn(request_body: str, content_type: str):
    """
    HTTP リクエストボディ -> Python オブジェクト変換

    - application/json のみ対応
    - 単一レコード / 複数レコードの両方を受け付ける
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    data = json.loads(request_body)

    # 単一 or 複数を DataFrame に正規化
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("JSON body must be an object or an array of objects")

    # 必須 feature が揃っているかチェック
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]):
    """
    前処理 + 推論を実行するコア関数
    """
    model: nn.Module = model_artifacts["model"]
    preprocess = model_artifacts["preprocess"]

    # 学習時と同じカラム順に揃える
    df = input_data[FEATURE_COLUMNS].copy()

    # sklearn 前処理
    X_proc = preprocess.transform(df)
    X_proc = X_proc.astype("float32")
    x_tensor = torch.from_numpy(X_proc)

    # 推論（勾配不要）
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy().tolist()

    # 複数件対応のため list で返す
    return preds


def output_fn(prediction, accept: str):
    """
    推論結果 -> HTTP レスポンス形式に変換
    """
    if accept != "application/json":
        # JSON のみサポート
        accept = "application/json"

    body = {"predictions": prediction}
    return json.dumps(body), accept
