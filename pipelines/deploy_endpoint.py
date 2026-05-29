import os
import yaml
import boto3
from sagemaker import get_execution_role
from sagemaker.model import ModelPackage

def deploy_latest_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    role = get_execution_role()
    sagemaker_client = boto3.client("sagemaker", region_name=config["aws"]["region"])
    
    # Query latest validated model package entry from the registry group
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=config["aws"]["model_package_group_name"],
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise ValueError(f"No approved models found inside group target: {config['aws']['model_package_group_name']}")
        
    latest_package_arn = packages[0]["ModelPackageArn"]
    print(f"Deploying approved model package resource: {latest_package_arn}")
    
    # Initialize deployment engine utilizing ModelPackage directly
    model = ModelPackage(
        role=role,
        model_package_arn=latest_package_arn
    )
    
    print("Spinning up managed endpoint infrastructure hosts...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=config["infrastructure"]["inference_instance"]
    )
    
    print(f"🎯 Managed Endpoint deployment successfully created: {predictor.endpoint_name}")
    return predictor.endpoint_name

if __name__ == "__main__":
    deploy_latest_model()
