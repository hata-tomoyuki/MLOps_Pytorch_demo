import os
import time

from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorchModel

# ===== .env 読み込み =====
load_dotenv()

# ===== 共通設定（環境変数から取得） =====
region = os.getenv("AWS_REGION", "ap-northeast-1")
role = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("S3_BUCKET")
base_prefix = os.getenv("S3_BASE_PREFIX", "sprint-weather")
instance_type = os.getenv("INFERENCE_INSTANCE_TYPE", "ml.m5.large")
endpoint_name = os.getenv("ENDPOINT_NAME", "sprint-weather-endpoint")

if not role:
    raise RuntimeError("SAGEMAKER_ROLE_ARN が環境変数で設定されていません。")
if not bucket:
    raise RuntimeError("S3_BUCKET が環境変数で設定されていません。")

# ===== S3 上のパス =====
# モデル: s3://{bucket}/{base_prefix}/model/model.tar.gz
model_data = f"s3://{bucket}/{base_prefix}/model/model.tar.gz"

# 推論コード: s3://{bucket}/{base_prefix}/code/inference/inference_code.tar.gz
source_s3 = f"s3://{bucket}/{base_prefix}/code/inference/inference_code.tar.gz"

# ===== イメージ URI =====
image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.3",
    py_version="py311",
    image_scope="inference",
    instance_type=instance_type,
)

# ===== モデル定義 =====
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    image_uri=image_uri,
    entry_point="inference.py",
    source_dir=source_s3,
    framework_version="2.3",
    py_version="py311",
)

# ===== エンドポイント作成（既存 endpoint_name を置き換え）=====
endpoint_config_name = f"{endpoint_name}-{int(time.time())}"

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)

print("Deployed endpoint:", endpoint_name)
