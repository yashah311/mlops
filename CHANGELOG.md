# Changelog

All notable changes to this project will be documented in this file. This project adheres to Semantic Versioning (`vMAJOR.MINOR.PATCH`).

---

## - 2026-05-30

### Added
- **Automated Processing Quality Gate:** Integrated `src/evaluate.py` via a SageMaker `ProcessingStep` to calculate validation accuracy against a standardized metrics JSON schema.
- **Conditional Workflow Routing:** Incorporated a native `ConditionStep` (`CheckAccuracyGate`) to automatically block model registration if the validation score falls below **80%**.
- **Centralized Parameter Management:** Added `config/pipeline_config.yaml` to decouple environment strings, infrastructure types, and global AWS configurations from runtime application logic.
- **Automated Integration Testing:** Added `tests/test_inference.py` to invoke the live endpoint with test payloads and handle success/failure signals via standard exit codes (`0` or `1`).
- **Asynchronous CI/CD Orchestration:** Configured `pipelines/run_pipeline.py` to use a fire-and-forget `PipelineSession` pattern, preventing the GitHub Actions build runner from blocking on long-running remote jobs.

### Fixed
- **SageMaker Container Deployment Error:** Re-architected the orchestration layout to use `ModelStep` combined with `SKLearnModel(env={...})`. This explicitly packs `inference.py` inside the model package manifest properties, completely eliminating the `AttributeError: 'NoneType' object has no attribute 'startswith'` error during `/ping` health checks.
- **Endpoint Config Immutability Blocker:** Modified `pipelines/deploy_endpoint.py` using raw `boto3` client properties to append unique timestamp suffixes to every configuration rollout, preventing name collision `ValidationException` errors.
- **Concurrent Deployment Lock Errors:** Built a defensive polling state loop inside the CD script that identifies an active `Updating` state, holding execution until the infrastructure lock releases before applying updates.
- **Self-Healing State Recovery:** Implemented an automated health check that identifies and deletes broken or `Failed` endpoints before launching a fresh deployment.

### Security Hardening
- **Least-Privilege Role Isolation:** Replaced the broad `AmazonSageMakerAdminIAMExecutionRole` administrative role with a dedicated, service-bound role named `SageMakerPipelineExecutionRole` containing scoped `AmazonSageMakerFullAccess` permissions.
- **Granular PassRole Boundaries:** Attached an inline policy override to the `github-actions-transformer` user account that explicitly allows passing only the specific pipeline worker role to the `://amazonaws.com` service principal, preventing privilege escalation exploits.
- **S3 Namespace Scoping:** Restricted programmatic data plane read/write permissions strictly to the target application bucket (`sagemaker-ap-south-1-194169602214`) and the auto-generated compiler bucket (`amazon-sagemaker-194169602214-ap-south-1-4z6615tqnogku9`).

### Dependencies Updated (`requirements.txt`)
- Enforced stability boundaries to guarantee compatibility across local runner nodes and cloud clusters:
  - `boto3>=1.34.0`
  - `sagemaker>=2.140.0,<3.0.0`
  - `pyyaml>=6.0`
