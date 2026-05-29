# Personal MLOps Project
This repository is a personal end-to-end MLOps demo showcasing model training, containerized model serving, and Kubernetes/Helm deployment.

## Project Summary
This project trains a breast cancer classification model, exposes it through a Flask scoring API, packages the service as a Docker image, and deploys it with Kubernetes and Helm.

## Skills Demonstrated
- Python model training with scikit-learn
- Flask REST API development
- Model serialization with `joblib`
- Docker containerization
- Kubernetes deployment manifests
- Helm chart authoring
- Seldon Core component templating

## Project Structure

- `py-flask-ml-score-api/`
  - `api.py` - Flask API for model scoring.
    - `/` returns a sample prediction for a hard-coded input.
    - `/predict` accepts POST JSON payloads and returns model predictions.
  - `model.py` - Trains an SVC classifier on the sklearn breast cancer dataset using `mean radius` and `mean concavity`.
  - `svc_model.model` - Serialized model artifact used by the API.
  - `Dockerfile` - Builds the Flask scoring service image and exposes port `5050`.
  - `py_flask_ml_score.yaml` - Kubernetes manifest for deploying the service and a LoadBalancer.

- `helm-ml-score-app/`
  - `Chart.yaml` - Helm chart metadata.
  - `values.yaml` - Default Helm configuration for app name, namespace, image, and ports.
  - `templates/` - Namespace, deployment, and service templates.

- `seldon-ml-score-component/`
  - `MLScore.py` - Seldon Core wrapper template for model serving.
  - `Dockerfile`, `Pipfile`, `Pipfile.lock` - runtime dependencies.

## What this project does

1. Trains a breast cancer classification model with scikit-learn.
2. Serializes the model to disk with `joblib`.
3. Builds a Flask API to serve predictions from the trained model.
4. Containers the service with Docker and exposes port `5050`.
5. Provides Kubernetes manifests and a Helm chart for deployment.
6. Includes a Seldon Core wrapper template for future production-grade serving.

## MLOps Lifecycle Mapping

| Lifecycle stage | Project component | Files / artifacts |
| --- | --- | --- |
| Data and experimentation | Dataset selection and feature engineering | `py-flask-ml-score-api/model.py` |
| Model training | Train and evaluate an SVC classifier | `py-flask-ml-score-api/model.py` |
| Model serialization | Save trained model for serving | `py-flask-ml-score-api/svc_model.model` |
| Model serving | REST API for prediction requests | `py-flask-ml-score-api/api.py` |
| Containerization | Package service as Docker image | `py-flask-ml-score-api/Dockerfile` |
| Deployment | Kubernetes manifest deployment | `py-flask-ml-score-api/py_flask_ml_score.yaml` |
| Deployment automation | Helm chart for repeatable deployment | `helm-ml-score-app/` |
| Model serving template | Seldon Core wrapper for future production | `seldon-ml-score-component/MLScore.py` |

## How to Use

### Run locally with Flask

```bash
cd py-flask-ml-score-api
pip install -r requirements.txt
python api.py
```

Send a request:

```bash
curl -X POST http://localhost:5050/predict -H 'Content-Type: application/json' -d '{"data": [13.77, 0.2344]}'
```

### Build Docker image

```bash
cd py-flask-ml-score-api
docker build -t mlops-score-api:latest .
```

### Deploy with Kubernetes manifest

```bash
kubectl apply -f py-flask-ml-score-api/py_flask_ml_score.yaml
```

### Deploy with Helm

```bash
cd helm-ml-score-app
helm install test-ml-score .
```

## Interview-Ready Highlights
- Built a complete MLOps workflow from model training to deployment.
- Demonstrated containerization and orchestration skills.
- Used a real dataset and a reproducible API serving pattern.
- Included a Helm chart and Kubernetes manifest for deployment automation.

## Future Improvements
- Add model evaluation metrics and validation.
- Extend `seldon-ml-score-component` to load the trained model.
- Add CI/CD automation for Docker image build and Kubernetes deployment.
- Add automated tests for the API and deployment manifests.

## Quick Links
* Jupyter notebook: https://colab.research.google.com/drive/1ydnBvjVr7oVX9y05JATdCmQJv1SWvLsB#scrollTo=GUmOP17g_0NR
* Install minikube: https://minikube.sigs.k8s.io/docs/start/
* Install helm: https://www.cyberithub.com/steps-to-install-helm-kubernetes-package-manager-on-linux/

