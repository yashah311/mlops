import boto3
import datetime
import time
import sagemaker

from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# -------------------------------------------------------------
# 1. Setup Environment
# -------------------------------------------------------------
session = sagemaker.Session()
role = get_execution_role()
sagemaker_client = boto3.client("sagemaker", region_name="ap-south-1")

model_package_group_name = "mlops-model-group"
model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# -------------------------------------------------------------
# 2. Define Pipeline Steps
# -------------------------------------------------------------
# Step A: Training
estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.0-1"
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator
)

# Step B: Model Registration (Inference Environment Variables Injected Here 🎯)
register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_group_name,
    approval_status="Approved",  # Set to approved directly to bypass manual console clicks
    
    # c:\Users\yash.shah\Downloads\mlops.ipynb
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    CRITICAL FIX: Forces the container registry payload to recognize inference.py
    env={
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": training_step.properties.ModelArtifacts.S3ModelArtifacts
    },
    tags=[
        {"Key": "project", "Value": "mlops"},
        {"Key": "version", "Value": model_version},
        {"Key": "env", "Value": "dev"}
    ]
)

# -------------------------------------------------------------
# 3. Create and Execute Pipeline
# -------------------------------------------------------------
pipeline = Pipeline(
    name="MLOpsPipelineVersioned",
    steps=[training_step, register_step],
    sagemaker_session=session
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f" Pipeline execution started. ARN: {execution.arn}")
print(f"Waiting for pipeline execution to finish...")
execution.wait()  # Synchronous wait block until training and registration are complete
print(" Pipeline execution completed successfully!")

# -------------------------------------------------------------
# 4. Extract Registered Model Artifacts
# -------------------------------------------------------------
approved_packages = sagemaker_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending"
)

if not approved_packages["ModelPackageSummaryList"]:
    raise ValueError(f"No approved model packages found in group: {model_package_group_name}")

latest_package = approved_packages["ModelPackageSummaryList"][0]
model_package_arn = latest_package["ModelPackageArn"]
print(f" Fetching Latest Registered Model Package: {model_package_arn}")

package_details = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
model_data_url = package_details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

# -------------------------------------------------------------
# 5. Deploy Real-Time Managed Endpoint
# -------------------------------------------------------------
print("Starting endpoint deployment...")
model = SKLearnModel(
    model_data=model_data_url,
    role=role,
    entry_point="inference.py",  # Make sure this local file is in your current directory folder
    framework_version="1.0-1"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
print(f"🎯 Managed Endpoint deployed successfully: {predictor.endpoint_name}")

# -------------------------------------------------------------
# 6. Test Model Inference Endpoint
# -------------------------------------------------------------
print("Sending test payload to the newly deployed endpoint...")
test_predictor = Predictor(
    endpoint_name=predictor.endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    sagemaker_session=session
)

sample_data = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189]]

inference_result = test_predictor.predict(sample_data)
print("\n Inference Result from SageMaker Endpoint:")
print(inference_result)
