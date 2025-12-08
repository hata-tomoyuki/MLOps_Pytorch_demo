import os
import urllib.parse

from dotenv import load_dotenv
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# ===== .env 読み込み =====
load_dotenv()  # カレントディレクトリの .env を読み込む

# ===== 共通設定（環境変数から取得） =====
region = os.getenv("AWS_REGION", "ap-northeast-1")
role = os.getenv("SAGEMAKER_ROLE_ARN")

bucket = os.getenv("S3_BUCKET")
base_prefix = os.getenv("S3_BASE_PREFIX", "sprint-weather")

# インスタンスタイプ
instance_type = os.getenv("TRAIN_INSTANCE_TYPE", "ml.m5.large")

# ハイパーパラメータ（型変換に注意）
hidden_dim = int(os.getenv("HIDDEN_DIM", "64"))
lr = float(os.getenv("LR", "0.001"))
max_epochs = int(os.getenv("MAX_EPOCHS", "50"))
batch_size = int(os.getenv("BATCH_SIZE", "64"))

# バリデーション（最低限）
if not role:
    raise RuntimeError("環境変数 SAGEMAKER_ROLE_ARN が設定されていません。")
if not bucket:
    raise RuntimeError("環境変数 S3_BUCKET が設定されていません。")

# ===== S3 パス（レイアウトに合わせて組み立て） =====
# 学習データ: s3://{bucket}/{base_prefix}/data/train/
train_s3_uri = f"s3://{bucket}/{base_prefix}/data/train/"

# トレーニングジョブ出力: s3://{bucket}/{base_prefix}/train_output/
output_path = f"s3://{bucket}/{base_prefix}/train_output/"

# 固定の本番モデルパス: s3://{bucket}/{base_prefix}/model/model.tar.gz
canonical_model_key = f"{base_prefix}/model/model.tar.gz"
canonical_model_uri = f"s3://{bucket}/{canonical_model_key}"

# ===== SageMaker セッション =====
session = sagemaker.Session()

# ===== Estimator 定義 =====
estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",  # train.py があるフォルダ
    role=role,
    framework_version="2.0.0",
    py_version="py310",
    instance_count=1,
    instance_type=instance_type,
    hyperparameters={
        "hidden_dim": hidden_dim,
        "lr": lr,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
    },
    sagemaker_session=session,
    output_path=output_path,
)

# ===== 学習実行 =====
estimator.fit(
    inputs={
        # ここが train.py の SM_CHANNEL_TRAIN に対応
        "train": train_s3_uri
    }
)

# ===== 学習済み model.tar.gz を「固定パス」にコピー =====
src_model_uri = estimator.model_data
print("Training job model_data:", src_model_uri)

parsed = urllib.parse.urlparse(src_model_uri)
src_bucket = parsed.netloc
src_key = parsed.path.lstrip("/")

s3 = boto3.client("s3", region_name=region)
s3.copy_object(
    Bucket=bucket,
    CopySource={"Bucket": src_bucket, "Key": src_key},
    Key=canonical_model_key,
)

print("Canonical model saved to:", canonical_model_uri)
