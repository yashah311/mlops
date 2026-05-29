import os
import yaml
import datetime
import sagemaker
import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel

def create_and_run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    target_region = config["aws"]["region"]
    print(f"Initializing SageMaker Session explicitly targeting region: {target_region}")
    
    # Instantiate clients with explicit region bindings to satisfy the SDK layer
    boto_session = boto3.Session(region_name=target_region)
    sagemaker_client = boto_session.client("sagemaker", region_name=target_region)
    
    custom_bucket = config["aws"].get("default_bucket")
    
    if custom_bucket and "YOUR_EXISTING_S3_BUCKET" not in custom_bucket:
        session = sagemaker.Session(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            default_bucket=custom_bucket
        )
    else:
        session = sagemaker.Session(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client
        )
    
    # Resolve the security context dynamically for target execution runners
    role = config["aws"].get("sagemaker_role_arn")
    if not role or "YOUR_SAGEMAKER_ROLE_NAME" in role:
        from sagemaker import get_execution_role
        role = get_execution_role()
        
    print(f"Executing workflow utilizing IAM Target Role: {role}")
    model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=config["infrastructure"]["training_instance"]
    )
    
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=os.path.join(base_dir, "src"),
        role=role,
        instance_type=instance_type_param,
        framework_version=config["infrastructure"]["framework_version"],
        sagemaker_session=session
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator
    )
    
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[config["infrastructure"]["inference_instance"]],
        transform_instances=[config["infrastructure"]["inference_instance"]],
        model_package_group_name=config["aws"]["model_package_group_name"],
        approval_status="Approved",
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": training_step.properties.ModelArtifacts.S3ModelArtifacts
        },
        tags=[
            {"Key": "project", "Value": config["metadata"]["project"]},
            {"Key": "version", "Value": model_version},
            {"Key": "env", "Value": config["metadata"]["env"]}
        ]
    )
    
    # Explicitly force the Pipeline definition to consume the regional session object
    pipeline = Pipeline(
        name=config["aws"]["pipeline_name"],
        parameters=[instance_type_param],
        steps=[training_step, register_step],
        sagemaker_session=session  
    )
    
    print("Uploading and synchronizing pipeline schema with AWS...")
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    print(f"🚀 Pipeline execution started! ARN: {execution.arn}")
    print("Waiting for training and model registration to complete...")
    execution.wait()
    print("✅ Pipeline executed successfully. Model package is registered.")

if __name__ == "__main__":
    create_and_run_pipeline()
