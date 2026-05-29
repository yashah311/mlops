import os
import yaml
import boto3
import datetime
from sagemaker.model import ModelPackage

def deploy_or_update_endpoint():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    role = config["aws"].get("sagemaker_role_arn")
    if not role or "YOUR_SAGEMAKER_ROLE_NAME" in role:
        from sagemaker import get_execution_role
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
        
    latest_package_arn = packages[0]["ModelPackageArn"]
    print(f"Found latest approved model package: {latest_package_arn}")
    
    # 2. Reference the Model Package for deployment
    model = ModelPackage(
        role=role,
        model_package_arn=latest_package_arn
    )
    
    # 3. Dynamic Naming Strategy 🎯
    endpoint_name = f"{config['aws']['model_package_group_name']}-prod"
    
    # Generate a unique config name using a timestamp to prevent ValidationException
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    
    print(f"Target Endpoint: {endpoint_name}")
    print(f"Creating Unique Config: {endpoint_config_name}")
    
    # 4. Deploy/Update the endpoint
    print("Orchestrating managed endpoint infrastructure. Please wait...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=config["infrastructure"]["inference_instance"],
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_config_name, # 🚀 CRITICAL FIX: Forces a unique configuration layer
        update_endpoint_with_new_model=True 
    )
    
    print(f"🎯 Production Endpoint successfully updated and active: {predictor.endpoint_name}")

if __name__ == "__main__":
    deploy_or_update_endpoint()
