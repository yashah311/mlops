import sagemaker
import datetime

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker import get_execution_role

# -------------------------------
# Setup
# -------------------------------

session = sagemaker.Session()
role = get_execution_role()

# ✅ Versioning
model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# -------------------------------
# Step 1: Training
# -------------------------------

estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",  # ✅ FIXED
    framework_version="1.0-1"
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator
)

# -------------------------------
# Step 2: Model Registration (Versioning ✅)
# -------------------------------

register_step = RegisterModel(
    name="RegisterModel",

    estimator=estimator,

    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,

    content_types=["application/json"],
    response_types=["application/json"],

    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],

    model_package_group_name="mlops-model-group",

    tags=[
        {"Key": "project", "Value": "mlops"},
        {"Key": "version", "Value": model_version},
        {"Key": "env", "Value": "dev"}
    ]
)

# -------------------------------
# Pipeline
# -------------------------------

pipeline = Pipeline(
    name="MLOpsPipelineVersioned",
    steps=[training_step, register_step],
    sagemaker_session=session
)

# -------------------------------
# Run pipeline
# -------------------------------

pipeline.upsert(role_arn=role)

execution = pipeline.start()

print("🚀 Pipeline execution started")
print("Execution ARN:", execution.arn)
print("Model Version:", model_version)