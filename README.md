# Enterprise End-to-End MLOps Pipeline on AWS SageMaker

An production-grade, asynchronous machine learning operations framework utilizing **Amazon SageMaker Pipelines** and **GitHub Actions** for secure, automated continuous integration and continuous delivery (CI/CD). 

This architecture enforces strict isolation between machine learning logic and cloud infrastructure management configurations, applies granular, least-privilege identity access management, guarantees zero-downtime rolling deployments, and incorporates automated evaluation gates.

---

## 🏗️ Architectural Topology & System Flow

```text
[GitHub Plane] ──(OIDC Token)──> [IAM Security Gateway]
                                          │
    ┌─────────────────────────────────────┴─────────────────────────────────────┐
    ▼                                                                           ▼
[pipelines/run_pipeline.py]                                        [pipelines/deploy_endpoint.py]
    │                                                                           │
    ▼                                                                           ▼
[SageMaker Pipeline Session Engine]                               [Boto3 Active Deployment Layer]
    │                                                                           │
    ├── Step 1: Training Job (train.py)                                         ├── Parse Registry Package Metadata
    ├── Step 2: Processing Evaluation (evaluate.py)                             ├── Generate Timestamped Configurations
    └── Step 3: Condition Check (Accuracy >= 80%)                               ├── Evaluate Infrastructure Health Status
               ├── Pass ──> [Register Model Package in Registry]                └── Execute Zero-Downtime Rolling Update
               └── Fail ──> [Halt Pipeline Lifecycle]                                   │
                                                                                        ▼
                                                                           [tests/test_inference.py]
                                                                           └── Smoke Test E2E Payload Validation
```

```mermaid
graph TD
    %% Styling and Classes
    classDef github fill:#24292e,stroke:#fff,stroke-width:1px,color:#fff;
    classDef iam fill:#e05243,stroke:#fff,stroke-width:1px,color:#fff;
    classDef sagemaker fill:#ff9900,stroke:#fff,stroke-width:1px,color:#fff;
    classDef storage fill:#3F8624,stroke:#fff,stroke-width:1px,color:#fff;

    %% GitHub Actions Build Plane
    subgraph GitPlane ["GitHub Actions Build Plane (CI/CD Pipeline)"]
        A[Git Push main]:::github --> B[Trigger Runner VM]:::github
        B --> C[Configure AWS OIDC/Credentials]:::github
        C --> D[Execute pipelines/run_pipeline.py]:::github
        C --> E[Execute pipelines/deploy_endpoint.py]:::github
        C --> F[Execute tests/test_inference.py]:::github
    end

    %% Identity & Access Boundary
    subgraph IAMBoundary ["AWS IAM Security Layer (Least Privilege)"]
        G[User: github-actions-transformer]:::iam
        H[Inline: GitHub-PassRole-Override]:::iam
        I[Role: SageMakerPipelineExecutionRole]:::iam
        
        G -. PassRole .-> I
    end

    %% Core Managed Compute Engine
    subgraph SMCompute ["Amazon SageMaker Orchestration Engine (DAG Workflow)"]
        D --> J[Upsert & Trigger Pipeline]:::sagemaker
        
        subgraph PipelineDAG ["Compiled DAG Architecture"]
            J --> K[Step 1: TrainModel <br> SKLearn Estimator]:::sagemaker
            K --> L[Step 2: EvaluateModel <br> SKLearn Processor]:::sagemaker
            L --> M{Step 3: CheckAccuracyGate <br> ConditionStep >= 80%}:::sagemaker
            M -- True --> N[Step 4: RegisterModelStep <br> ModelStep]:::sagemaker
            M -- False --> O[Terminate Pipeline Execution]:::sagemaker
        end
    end

    %% Data Plane & Artifact Tracking
    subgraph DataPlane ["Storage & Asset Management Layer"]
        P[(S3 Bucket <br> sagemaker-ap-south-1-194169602214)]:::storage
        Q[(S3 Bucket <br> amazon-sagemaker-...-4z6615tqnogku9)]:::storage
        R[SageMaker Model Registry <br> group: mlops-model-group]:::sagemaker

        K <--> |Read Input / Write Model| P
        L <--> |Read Model / Write metrics.json| P
        J <--> |DAG Blueprint Schema| Q
        N --> |Approved Version Manifest <br> with Customer Metadata| R
    end

    %% Continuous Delivery Pipeline
    subgraph LiveServing ["Live Serving Plane (Real-Time Endpoint)"]
        E --> S[Query Latest Approved Model Package]:::sagemaker
        R -. Read Metadata .-> S
        S --> T[Boto3 Client Wrapper Engine]:::sagemaker
        T --> |Dynamic Timestamp Config| U[Create EndpointConfig]:::sagemaker
        U --> V{DescribeEndpoint Status}:::sagemaker
        
        V -- Failed --> W[Delete Broken Endpoint]:::sagemaker
        W --> X[Create New Endpoint]:::sagemaker
        
        V -- InService / Updating --> Y[Polled Wait Loop]:::sagemaker
        Y --> Z[Update Endpoint <br> Zero-Downtime Rolling Update]:::sagemaker
    end

    %% Verification Gate
    subgraph QualityVerification ["Production Integration Testing Gate"]
        F --> AA[Invoke Endpoint Client Request]:::github
        AA --> |Secure Matrix Transmission| X
        AA --> |Secure Matrix Transmission| Z
        X --> |JSON Inference Vector Response| AB[Assert Array Validation Code: 0]:::github
        Z --> |JSON Inference Vector Response| AB
    end
```


---

## ⚡ Key Architectural Features & Design Choices

### 1. Decoupled Code & Infrastructure Configuration
ML application logic resides strictly inside the `src/` directory, while deployment configurations are parameterised using a centralized configuration model (`config/pipeline_config.yaml`). This pattern ensures clean environments, mitigates dependency drift, and allows for seamless changes to infrastructure types without touching execution code.

### 2. Resolution of the SageMaker `/ping` Root Defect
Standard SageMaker pipeline registrations (`RegisterModel`) pull raw output files directly from the execution file layers, stripping away deployment handler dependencies. 

This pipeline design re-architects this layout to implement the `sagemaker.workflow.model_step.ModelStep` combined with the `PipelineSession` framework. By injecting serving variables (`SAGEMAKER_PROGRAM`) natively inside the `SKLearnModel` definition, configuration data maps correctly inside the registry metadata. This ensures that the Gunicorn processes locate the custom execution modules instantly and clears the `/ping` container health check error.

### 3. Native Processing & Condition Evaluation Gates
A dedicated `ProcessingStep` extracts performance metrics into a standardized JSON report on S3. A subsequent `ConditionStep` handles routing logic: model metadata packages are only approved and moved to the registry if performance meets or exceeds the required threshold (e.g., 80% accuracy).

### 4. Dynamic Rolling Deployments with Boto3 Orchestration
Rather than using high-level SDK abstractions that obscure infrastructure dependencies, `pipelines/deploy_endpoint.py` uses raw `boto3` parameters. This configuration implements an enterprise-ready continuous delivery flow:
* **Config Immutability Handling:** Every deployment run generates an immutable, timestamped endpoint configuration to prevent naming collisions.
* **Resilient State Management:** The deployment layer checks the active health status of existing infrastructure before running changes. It automatically deletes failed resources to clear the workspace and polls busy updates smoothly.
* **Zero-Downtime Blue/Green Updates:** Updates are sent to the production endpoints as managed rolling updates, shifting incoming requests cleanly across active hosts without down-time.

---

## 🛠️ Pipeline Implementation & Visual Verification

### Step 1: GitHub Actions Continuous Integration Trigger
Every code change merged into the target branch starts the workflow runner. The workflow securely assumes roles through an OpenID Connect (OIDC) connection token without requiring persistent AWS Access Keys.

<!-- PLACEHOLDER: Upload a screenshot of your successful GitHub Actions build execution steps here -->
![GitHub Actions Workflow Pipeline Run](docs/images/01_gha_workflow_run.png)

### Step 2: Managed Orchestration via SageMaker Pipelines
The SageMaker orchestration engine converts your Python scripts into a directed execution workflow. 

<!-- PLACEHOLDER: Upload a screenshot from your AWS SageMaker Studio Console showing the completed pipeline execution graph here -->
![SageMaker Pipeline Completed DAG](docs/images/02_sagemaker_pipeline_dag.png)

### Step 3: Model Lineage Tracking & Registry Storage
Models that surpass the performance gate threshold appear in the central model registry group. This records execution parameters, evaluation values, and validation matrices.

<!-- PLACEHOLDER: Upload a screenshot of your AWS SageMaker Model Registry console showing the approved versions here -->
![SageMaker Model Registry Approved Metrics](docs/images/03_model_registry.png)

### Step 4: Real-Time Blue/Green Inference Endpoint Deployment
The deployment engine acts as a continuous delivery worker. It references structural registry values, processes live traffic updates, and changes infrastructure settings with zero system downtime.

<!-- PLACEHOLDER: Upload a screenshot of your active AWS SageMaker real-time inference endpoint configuration here -->
![SageMaker Active Serving Endpoint](docs/images/04_active_endpoint.png)

---

## 🔒 Security Hardening & IAM Governance

This infrastructure enforces a zero-trust model. The automated GitHub Actions runner authenticates using scoped programmatic access configurations, blocking access to unrelated account components.

### 1. Minimal-Access Programmatic User Policy
Attach this policy directly to your automation user (`github-actions-transformer`) to grant the specific pipeline management capabilities required for compilation and orchestration tasks:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "SageMakerPipelineAndEndpointManagement",
			"Effect": "Allow",
			"Action": [
				"sagemaker:CreateTrainingJob",
				"sagemaker:DescribeTrainingJob",
				"sagemaker:CreateProcessingJob",
				"sagemaker:DescribeProcessingJob",
				"sagemaker:CreateModel",
				"sagemaker:DescribeModel",
				"sagemaker:DeleteModel",
				"sagemaker:CreateModelPackage",
				"sagemaker:DescribeModelPackage",
				"sagemaker:ListModelPackages",
				"sagemaker:CreateModelPackageGroup",
				"sagemaker:DescribeModelPackageGroup",
				"sagemaker:ListModelPackageGroups",
				"sagemaker:CreateEndpointConfig",
				"sagemaker:DescribeEndpointConfig",
				"sagemaker:DeleteEndpointConfig",
				"sagemaker:CreateEndpoint",
				"sagemaker:DescribeEndpoint",
				"sagemaker:UpdateEndpoint",
				"sagemaker:DeleteEndpoint",
				"sagemaker:ListEndpoints",
				"sagemaker:CreatePipeline",
				"sagemaker:UpdatePipeline",
				"sagemaker:StartPipelineExecution",
				"sagemaker:DescribePipeline",
				"sagemaker:DescribePipelineExecution",
				"sagemaker:ListPipelineExecutions",
				"sagemaker:AddTags",
				"sagemaker:InvokeEndpoint"
			],
			"Resource": [
				"arn:aws:sagemaker:ap-south-1:194169602214:training-job/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:processing-job/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:model/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:model-package-group/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:model-package/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:endpoint-config/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:endpoint/*",
				"arn:aws:sagemaker:ap-south-1:194169602214:pipeline*"
			]
		},
		{
			"Sid": "S3BucketAndArtifactAccess",
			"Effect": "Allow",
			"Action": [
				"s3:GetObject",
				"s3:PutObject",
				"s3:ListBucket",
				"s3:DeleteObject",
				"s3:GetBucketLocation"
			],
			"Resource": [
				"arn:aws:s3:::sagemaker-ap-south-1-194169602214",
				"arn:aws:s3:::sagemaker-ap-south-1-194169602214/*",
				"arn:aws:s3:::amazon-sagemaker-194169602214-ap-south-1-4z6615tqnogku9",
				"arn:aws:s3:::amazon-sagemaker-194169602214-ap-south-1-4z6615tqnogku9/*"
			]
		}
	]
}
```

### 2. User Inline Policy: Secure Role Delegation Cross-Boundary
To protect against privilege escalation attacks, AWS explicitly prevents standard IAM users from configuring or assigning execution capabilities that exceed their own security boundaries. 

This project solves this by attaching a granular `iam:PassRole` inline block to the programmatic user account. This explicitly allows it to pass only the specific, scoped service role (`SageMakerPipelineExecutionRole`) containing `AmazonSageMakerFullAccess` to the underlying compute instances, while explicitly restricting it from passing broader administrative credentials:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "DirectGitHubPassRoleOverride",
			"Effect": "Allow",
			"Action": "iam:PassRole",
			"Resource": [
				"arn:aws:iam::194169602214:role/service-role/SageMakerPipelineExecutionRole"
			],
			"Condition": {
				"StringEquals": {
					"iam:PassedToService": "://amazonaws.com"
				}
			}
		}
	]
}
```

---

## 🛠️ Operational Deployment & Execution Guide

### Local Evaluation & Interactive Testing
To run workflow steps manually or inspect performance characteristics inside an integrated development workspace (such as SageMaker Studio Notebook cells):

```bash
# Initialize project requirements
pip install -r requirements.txt

# Compile, upsert, and run the pipeline configuration asynchronously
python pipelines/run_pipeline.py

# Poll registry changes and trigger a rolling update deployment
python pipelines/deploy_endpoint.py

# Execute smoke validation scripts against active endpoints
python tests/test_inference.py
```

### Git Automation Flow
The active configuration uses a hands-free integration path triggered on code changes to the primary repository branch. To release changes:

```bash
git add .
git commit -m "feat: incorporate end-to-end operational code refactoring fixes"
git push origin main
```
Track live build container tracking metrics and workflow progress directly under your repository's **GitHub Actions Console Dashboard**.
