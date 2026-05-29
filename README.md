# Production SageMaker MLOps Pipeline 🚀

An enterprise-grade, end-to-end MLOps pipeline running on **AWS SageMaker** and automated via **GitHub Actions**. This architecture cleanly separates core ML logic from infrastructure orchestration, handles automated model versioning via the SageMaker Model Registry, and safely deploys real-time endpoints without running into common runtime container initialization errors.

---

## 🏗️ Architecture Layout

The codebase uses a production-ready directory structure designed to prevent dependency pollution and ensure local code components easily bind with remote cloud resources:

```text
yashah311-mlops/
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions CI/CD automation workflow
├── config/
│   └── pipeline_config.yaml# Centralized infrastructure & environment parameters
├── src/                    # 📦 Core ML Scripts (Runs INSIDE SageMaker containers)
│   ├── train.py            # Model training & model artifact packaging script
│   └── inference.py        # Custom serving handler (Prevents /ping failure errors)
├── pipelines/              # 🚀 Cloud Infrastructure Orchestration scripts
│   ├── run_pipeline.py     # Compiles & executes the SageMaker pipeline
│   └── deploy_endpoint.py  # Fetches approved models & triggers live deployments
├── tests/                  # 🧪 Quality Verification Gates
│   └── test_inference.py   # Integration matrix smoke tests against the live endpoint
└── requirements.txt        # Host orchestration execution requirements
```

---

## ⚡ Key Features

* **Anti-Drift Script Packaging:** `src/train.py` dynamically embeds `inference.py` directly inside the compressed `model.tar.gz` archive during training execution. This eliminates the standard SageMaker `AttributeError: 'NoneType' object has no attribute 'startswith'` during `/ping` health checks.
* **Unified Pipeline Automation:** Compiles multi-step execution schemas (`TrainingStep` → `RegisterModel`) into a parameterised Directed Acyclic Graph (DAG) using the SageMaker Python SDK.
* **Continuous Delivery:** Automatic blue/green endpoint updates using `update_endpoint_with_new_model=True` safely shifts live production traffic to newly registered and approved models.
* **Zero-Cost CI/CD Build Plane:** Orchestration layers are triggered using GitHub Actions runners, leveraging standard AWS API keys isolated securely inside repository secrets.

---

## 🛠️ Infrastructure Configuration

Environment variables, instance scales, and account resource naming targets are managed inside `config/pipeline_config.yaml` to ensure portability:

```yaml
aws:
  region: "ap-south-1"
  model_package_group_name: "mlops-model-group"
  pipeline_name: "MLOpsPipelineVersioned"

infrastructure:
  training_instance: "ml.m5.large"
  inference_instance: "ml.m5.large"
  framework_version: "1.0-1"

metadata:
  project: "mlops"
  env: "dev"
```

---

## 🚀 Execution Guide

### 1. Local / Workspace Testing
To execute the pipeline orchestration workflow or trigger the deployment runner manually within your SageMaker Studio or local environment:

```bash
# Install local dependencies
pip install -r requirements.txt

# Compile and start the SageMaker training pipeline
python pipelines/run_pipeline.py

# Deploy the latest approved model registry artifact to an endpoint
python pipelines/deploy_endpoint.py

# Run integration tests against the live endpoint
python tests/test_inference.py
```

### 2. Automated Git-Triggered Execution
To use automated continuous integration, push your changes to your remote repository. The pipeline will execute through GitHub Actions:

1. Commit your updated code blocks to your repository.
2. Push your changes directly to the target branch:
   ```bash
   git add .
   git commit -m "feat: implement structural cloud infrastructure updates"
   git push origin main
   ```
3. Monitor your active step progress logs inside the **Actions** tab on your GitHub repository page.

---

## 🔒 Security Hardening

The automated orchestration runner connects to AWS using an IAM user configured with least-privilege access permissions. This configuration restricts access to your specified SageMaker resources, specific S3 buckets, and includes strict condition blocks for role passing:

* **`AWS_ACCESS_KEY_ID`**: Secure deployment access key identifier.
* **`AWS_SECRET_ACCESS_KEY`**: Protected secret string value.

*Never check raw programmatic account keys directly into your version control repository files.*
