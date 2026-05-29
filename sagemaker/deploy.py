import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Model location (after training job completes)
model_data = "s3://amazon-sagemaker-194169602214-ap-south-1-4z6615tqnogku9/shared/sagemaker-scikit-learn-2026-05-29-09-22-54-482/output/model.tar.gz"

model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    framework_version="1.0-1"
)

predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1
)

print("Model deployed! Endpoint name:", predictor.endpoint_name)
