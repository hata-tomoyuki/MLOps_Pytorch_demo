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


def _patch_onehot_categories_no_nan(preprocess: Any) -> None:
    """
    OneHotEncoder.categories_ から、欠損値（NaN / None / pandas.NA）のみを安全に除去する。
    文字列カテゴリなどは除去しない。
    """
    import pandas as pd

    def is_invalid(v):
        # None / pd.NA
        if v is None or v is pd.NA:
            return True
        # float NaN
        if isinstance(v, float) and np.isnan(v):
            return True
        # それ以外（文字列など）は有効カテゴリとして残す
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
        if isinstance(obj, OneHotEncoder):
            _patch_encoder(obj)
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                _walk(step)
        if isinstance(obj, ColumnTransformer):
            transformers_list = getattr(obj, "transformers_", None) or getattr(obj, "transformers", [])
            for _, transformer, _ in transformers_list:
                _walk(transformer)

    _walk(preprocess)


# 学習時の feature_cols と同じ順序で扱う
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


# ===== SageMaker 必須: model_fn / input_fn / predict_fn / output_fn =====

def model_fn(model_dir: str):
    """
    コンテナ起動時に 1 回だけ呼ばれて、
    /opt/ml/model 以下から学習済みモデルと前処理を読み込む。
    """
    print(f"[model_fn] model_dir = {model_dir}")
    files = os.listdir(model_dir)
    print(f"[model_fn] files in model_dir: {files}")

    model_path = os.path.join(model_dir, "model.bin")
    preprocess_path = os.path.join(model_dir, "preprocess.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(f"model.pt not found in {model_dir}")
    if not os.path.exists(preprocess_path):
        raise RuntimeError(f"preprocess.pkl not found in {model_dir}")

    # 前処理パイプラインをロード
    preprocess = joblib.load(preprocess_path)
    print("[model_fn] preprocess.pkl loaded")

    # OneHotEncoder の categories_ から欠損のみ除去
    _patch_onehot_categories_no_nan(preprocess)
    print("[model_fn] preprocess patched (removed NaN/None from OneHotEncoder.categories_)")

    # ===== モデル checkpoint のロード =====
    ckpt = torch.load(model_path, map_location="cpu")

    # 推奨形式: {"model_state_dict", "n_features", "hidden_dim"}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        n_features = int(ckpt["n_features"])
        hidden_dim = int(ckpt.get("hidden_dim", 64))
        print(f"[model_fn] loaded from new-format ckpt: n_features={n_features}, hidden_dim={hidden_dim}")
    else:
        # 後方互換用: 旧形式（state_dict だけを保存していた場合など）
        state_dict = ckpt

        # Lightning の checkpoint 形式 {"state_dict": {...}} にも対応
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if "0.weight" in state_dict:
            # core_model.state_dict() をそのまま保存していたパターン
            first_linear_weight = state_dict["0.weight"]
        elif "model.0.weight" in state_dict:
            # LightningModule 全体の state_dict を保存していたパターン
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

        n_features = first_linear_weight.shape[1]
        hidden_dim = 64
        print(f"[model_fn] backward-compat: inferred n_features={n_features}, hidden_dim={hidden_dim}")

    # ===== nn.Sequential としてモデルを構築 =====
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

    return {"model": model, "preprocess": preprocess}


def input_fn(request_body: str, content_type: str):
    """
    HTTP リクエストボディ -> Python オブジェクト（ここでは pandas.DataFrame）
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    data = json.loads(request_body)

    # 単一レコード or 複数レコード両対応
    # 例:
    #   単一: {"sex":"M", "age":25, ...}
    #   複数: [{"sex":"M", "age":25, ...}, {...}, ...]
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("JSON body must be an object or an array of objects")

    # カラム順の確認（足りないカラムがあればエラー）
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]):
    """
    前処理 + 推論
    input_data: input_fn が作った DataFrame
    model_artifacts: model_fn が返した dict
    """
    model: nn.Module = model_artifacts["model"]
    preprocess = model_artifacts["preprocess"]

    # FEATURE_COLUMNS の順序で並べ替える（学習時と同じ順序）
    df = input_data[FEATURE_COLUMNS].copy()

    # scikit-learn の前処理で transform
    X_proc = preprocess.transform(df)  # numpy array (N, n_features)
    X_proc = X_proc.astype("float32")
    x_tensor = torch.from_numpy(X_proc)

    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy().tolist()

    # 複数件に備えてリストのまま返す
    return preds


def output_fn(prediction, accept: str):
    """
    推論結果 -> HTTP レスポンスボディ
    """
    if accept != "application/json":
        # JSON だけサポート
        accept = "application/json"

    # prediction は [y1, y2, ...] の形を想定
    body = {"predictions": prediction}
    return json.dumps(body), accept
