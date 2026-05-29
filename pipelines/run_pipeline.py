import os
import yaml
import datetime
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel

def create_and_run_pipeline():
    # Read modular environment runtime configuration parameters
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    session = sagemaker.Session()
    role = get_execution_role()
    
    model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Construct base estimator framework tracking definition
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=os.path.join(base_dir, "src"), # Packages everything in src/ automatically
        role=role,
        instance_type=config["infrastructure"]["training_instance"],
        framework_version=config["infrastructure"]["framework_version"],
        sagemaker_session=session
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator
    )
    
    # Create workflow registration configurations matching manifest hooks
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
        # Keep environment configuration keys attached to registry entries
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
    
    pipeline = Pipeline(
        name=config["aws"]["pipeline_name"],
        steps=[training_step, register_step],
        sagemaker_session=session
    )
    
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    print(f"🚀 SageMaker Pipeline compilation successfully synchronized.")
    print(f"Execution ARN identifier: {execution.arn}")
    print(f"Version payload marker: {model_version}")
    return execution

if __name__ == "__main__":
    create_and_run_pipeline()
