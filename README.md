# Sprint Weather ML Pipeline — README

このプロジェクトは、**SageMaker 上での機械学習モデルの自動学習・デプロイ基盤**です。

以下の構成により、

1. 学習コード（train）
2. 推論コード（inference）
3. 学習データ
4. 本番モデルアーティファクト
5. トレーニング出力

を一つの S3 バケット配下に整理し、再現性と拡張性の高い ML パイプラインを構築します。

---

## 📁 S3 ディレクトリ構成

```text
s3://sagemaker-ap-northeast-1-147367797159/sprint-weather/
├── train/                         # 学習用 CSV
│     └── training-data.csv
│
├── code/
│     ├── train/train_code.tar.gz          # train.py + requirements.txt
│     └── inference/inference_code.tar.gz  # inference.py + requirements.txt
│
├── train_output/                  # SageMaker の学習ジョブ出力（自動生成）
│     └── pytorch-training-XXXX/output/model.tar.gz
│
└── model/
      └── model.tar.gz             # 最新版モデル（毎回上書き）
```

---

# 🚀 1. 準備：コードを tar.gz にまとめて S3 へアップロード

## 1-1. 学習コード（train_code.tar.gz）

対象ファイル：

```
train.py
requirements.txt
```

作成とアップロード：

```bash
tar -czvf train_code.tar.gz train.py requirements.txt

aws s3 cp train_code.tar.gz \
  s3://sagemaker-ap-northeast-1-147367797159/sprint-weather/code/train/
```

---

## 1-2. 推論コード（inference_code.tar.gz）

対象ファイル：

```
inference.py
requirements.txt
```

作成とアップロード：

```bash
tar -czvf inference_code.tar.gz inference.py requirements.txt

aws s3 cp inference_code.tar.gz \
  s3://sagemaker-ap-northeast-1-147367797159/sprint-weather/code/inference/
```

---

# 🏋️ 2. 学習ジョブを実行（run_training.py）

ローカルで以下を実行します：

```bash
python run_training.py
```

## ▼ run_training.py が実施すること

1. SageMaker トレーニングジョブを起動
2. 入力データ `sprint-weather/train/` を読み込み
3. 学習成果物（model.pt + preprocess.pkl）を
   SageMaker が `train_output/.../output/model.tar.gz` に出力
4. その model.tar.gz を自動で以下にコピー：

```
s3://sagemaker-ap-northeast-1-147367797159/sprint-weather/model/model.tar.gz
```

→ このファイルが **本番用最新モデル** として常に最新化されます。

---

# 🚀 3. 推論エンドポイントをデプロイ（deploy.py）

```bash
python deploy.py
```

## ▼ deploy.py が実施すること

1. S3 の最新モデル（`model/model.tar.gz`）をロード
2. 推論コード（inference_code.tar.gz）をロード
3. SageMaker エンドポイント `sprint-weather-endpoint` を新規作成
   ※既存の同名エンドポイントがある場合は **削除してから実行する必要があります**

---

# 🧹 エンドポイント更新時の注意

エンドポイント名を固定（推奨）する場合は、
**再デプロイ前に必ず削除してください。**

```bash
aws sagemaker delete-endpoint \
  --endpoint-name sprint-weather-endpoint
```

削除完了（Deleted）になったあとで：

```bash
python deploy.py
```

---

# 🔁 運用フロー（再学習 → 再デプロイ）

1. 新しい学習 CSV を S3 `train/` に追加
2. `python run_training.py` を実行
   → 最新モデルが `model/model.tar.gz` に上書きされる
3. 旧エンドポイント削除
4. `python deploy.py` を実行
   → 最新モデルでエンドポイント再構築
---
