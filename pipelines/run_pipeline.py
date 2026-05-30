import os
import yaml
import datetime
import sagemaker
import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
# 🎯 KEY UPDATE: Switch to modern explicit Model and Step declarations
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep

def create_and_run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    target_region = config["aws"]["region"]
    print(f"Initializing SageMaker explicit session workspace targeting: {target_region}")
    
    boto_session = boto3.Session(region_name=target_region)
    sagemaker_client = boto_session.client("sagemaker", region_name=target_region)
    
    custom_bucket = config["aws"].get("default_bucket")
    session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=custom_bucket if custom_bucket and "YOUR_EXISTING_S3" not in custom_bucket else None
    )
    
    role = config["aws"].get("sagemaker_role_arn")
    print(f"Orchestrating workflow utilizing Execution Role: {role}")
    model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=config["infrastructure"]["training_instance"]
    )
    
    # Standard Model Training Definition Step
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
    
    # 🚀 THE FIX: Create an explicit Model object that links code straight to the registry artifact
    model_instance = SKLearnModel(
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=os.path.join(base_dir, "src"), # 🎯 Forces SageMaker to bundle inference.py into the package
        framework_version=config["infrastructure"]["framework_version"],
        sagemaker_session=session
    )
    
    # Wrap the Model object inside a proper register pipeline workflow step
    register_step = ModelStep(
        name="RegisterModelStep",
        step_args=model_instance.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[config["infrastructure"]["inference_instance"]],
            transform_instances=[config["infrastructure"]["inference_instance"]],
            model_package_group_name=config["aws"]["model_package_group_name"],
            approval_status="Approved"
        )
    )
    
    pipeline = Pipeline(
        name=config["aws"]["pipeline_name"],
        parameters=[instance_type_param],
        steps=[training_step, register_step],
        sagemaker_session=session  
    )
    
    print("Uploading and synchronizing updated pipeline schema with AWS...")
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    print(f"🚀 Pipeline execution started! ARN: {execution.arn}")
    print("Waiting for training and explicit model registration to complete...")
    execution.wait()
    print("✅ Pipeline executed successfully. Model package version is registered with custom inference handlers.")

if __name__ == "__main__":
    create_and_run_pipeline()
