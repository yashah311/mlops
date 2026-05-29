import os
import yaml
import boto3
from sagemaker.session import get_execution_role
from sagemaker.model import ModelPackage

def deploy_or_update_endpoint():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 🎯 FIX: Safely resolve the role ARN for the deployment layer
    try:
        role = config["aws"]["sagemaker_role_arn"]
        if not role or "YOUR_SAGEMAKER_ROLE_NAME" in role:
            raise ValueError
    except (KeyError, ValueError):
        role = get_execution_role()
        
    sagemaker_client = boto3.client("sagemaker", region_name=config["aws"]["region"])
    
    # 1. Fetch the latest approved model package from the Registry Group
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=config["aws"]["model_package_group_name"],
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise ValueError(f"No approved model packages found in group: {config['aws']['model_package_group_name']}")
        
    latest_package_arn = packages[0]["ModelPackageArn"] # Fix index selection syntax
    print(f"Found latest approved model package: {latest_package_arn}")
    
    # 2. Reference the Model Package for deployment
    model = ModelPackage(
        role=role,
        model_package_arn=latest_package_arn
    )
    
    # Standardized endpoint naming strategy
    endpoint_name = f"{config['aws']['model_package_group_name']}-prod"
    print(f"Targeting deployment endpoint name: {endpoint_name}")
    
    # 3. Deploy the model
    print("Spinning up managed endpoint infrastructure. Please wait...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=config["infrastructure"]["inference_instance"],
        endpoint_name=endpoint_name,
        update_endpoint_with_new_model=True 
    )
    
    print(f"🎯 Production Endpoint deployed successfully: {predictor.endpoint_name}")

if __name__ == "__main__":
    deploy_or_update_endpoint()
