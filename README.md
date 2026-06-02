# 🚀 Production-Grade MLOps Pipeline on AWS (SageMaker + GitHub Actions)

A fully automated, enterprise-grade MLOps system that orchestrates the complete machine learning lifecycle — from code commit to production deployment — using Amazon SageMaker Pipelines and GitHub Actions.

## 📌 TL;DR
- Push to main → model trains automatically
- Model promoted only if evaluation ≥ 80%
- Zero-downtime endpoint deployment
- Secure OIDC-based authentication
- Runs within AWS free-tier credits

---

## 🏗️ Architecture Overview

GitHub Actions → SageMaker Pipeline → Model Registry → Endpoint → Inference Test

---

## ⚙️ CI/CD Pipeline

1. Configure AWS via OIDC
2. Run pipeline:
   python pipelines/run_pipeline.py
3. Deploy model:
   python pipelines/deploy_endpoint.py
4. Test endpoint:
   python tests/test_inference.py

---

## 🔬 Pipeline Steps

### Training
- src/train.py

### Evaluation
Outputs JSON metrics like:
{
  "accuracy": 0.84
}

### Condition Gate
- Only promotes if accuracy ≥ 80%

### Registry
- Versioned model storage

---

## 🚀 Deployment

- Immutable endpoint configs
- Blue/Green deployment
- Self-healing logic

---

## 🧪 Validation

Example output:
{
  "prediction": [0.87]
}

---

## ☁️ AWS Services

- SageMaker
- S3
- IAM
- Model Registry

---

## 🔐 Security

- OIDC authentication
- Least privilege IAM
- Scoped role delegation

---

## 📁 Structure

pipelines/
src/
tests/
config/
.github/workflows/

---

## 💸 Cost Optimization

- Uses free-tier resources
- All resources deleted after testing

---

## 🧹 Cleanup

- Delete endpoints
- Delete S3 buckets
- Delete models
- Remove IAM roles

---

## 🎯 Outcome

Demonstrates production-grade MLOps with automation, security, and cost efficiency.
