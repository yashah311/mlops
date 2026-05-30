import os
import yaml
import boto3
import datetime
import time

def deploy_or_update_endpoint():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    role = config["aws"].get("sagemaker_role_arn")
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
    
    # 2. Naming Layout Definitions
    endpoint_name = f"{config['aws']['model_package_group_name']}-dev-endpoint"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    model_name = f"{config['aws']['model_package_group_name']}-model-{timestamp}"
    
    print(f"Creating Model Resource: {model_name}")
    print(f"Creating Unique Config: {endpoint_config_name}")
    print(f"Targeting Endpoint: {endpoint_name}")
    
    # 3. Create SageMaker Model Entity
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "ModelPackageName": latest_package_arn
        }
    )
    
    # 4. Create Endpoint Configuration
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": config["infrastructure"]["inference_instance"],
                "InitialVariantWeight": 1.0
            }
        ],
        Tags=[
            {"Key": "project", "Value": config["metadata"]["project"]},
            {"Key": "env", "Value": config["metadata"]["env"]}
        ]
    )
    
    # 5. Smart Health Check Logic 🎯
    endpoint_exists = False
    should_delete_failed = False
    
    try:
        desc_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        status = desc_response["EndpointStatus"]
        print(f"Current Endpoint Status located: {status}")
        
        if status == "Failed":
            print("⚠️ Endpoint is in a failed state. Flagging for removal...")
            should_delete_failed = True
            
    except sagemaker_client.exceptions.ClientError:
        print("Endpoint does not exist yet. Preparing first-time build sequence...")
        
    # 6. Execution Branching Handling
    if should_delete_failed:
        print(f"Deleting failed endpoint '{endpoint_name}' to clear workspace...")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Wait until the deletion is fully completed on AWS
        print("Waiting for deletion confirmation status...")
        while True:
            try:
                sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(10)
            except sagemaker_client.exceptions.ClientError:
                print("✅ Failed endpoint completely purged.")
                break
        endpoint_exists = False # Reset flag to force a fresh creation run
        
    if endpoint_exists:
        print("Performing seamless rolling update on active endpoint tracking layer...")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print("Initializing clean first-time endpoint generation run...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
    print(f"🚀 MLOps Endpoint orchestrator finished monitoring initialization for: {endpoint_name}")

if __name__ == "__main__":
    deploy_or_update_endpoint()
