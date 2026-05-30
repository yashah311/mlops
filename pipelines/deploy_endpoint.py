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
    
    # 1. Fetch latest approved model package
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=config["aws"]["model_package_group_name"],
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise ValueError(f"No approved packages found.")
        
    latest_package_arn = packages[0]["ModelPackageArn"]
    print(f"Found latest approved model package: {latest_package_arn}")
    
    # 2. Dynamic Naming Definitions
    endpoint_name = config["aws"]["endpoint_name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    model_name = f"{config['aws']['model_package_group_name']}-model-{timestamp}"
    
    # 3. Create SageMaker Model Entity
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"ModelPackageName": latest_package_arn}
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
        ]
    )
    
    # 5. Handle In-Progress Updates Natively 🎯
    endpoint_exists = False
    should_delete_failed = False
    
    try:
        while True:
            desc_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_exists = True
            status = desc_response["EndpointStatus"]
            print(f"Current Endpoint Status: {status}")
            
            if status == "Updating":
                print("⏳ Endpoint is busy updating. Waiting 30 seconds for lock to clear...")
                time.sleep(30)
                continue
            elif status == "Failed":
                print("⚠️ Endpoint failed. Marking for removal...")
                should_delete_failed = True
                break
            else:
                break # Status is InService and clear to update
                
    except sagemaker_client.exceptions.ClientError:
        print("Endpoint does not exist yet. Proceeding with fresh deployment.")
        
    # 6. Execution Paths
    if should_delete_failed:
        print(f"Purging failed endpoint...")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        while True:
            try:
                sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(10)
            except sagemaker_client.exceptions.ClientError:
                break
        endpoint_exists = False
        
    if endpoint_exists:
        print(f"🚀 Performing rolling update to config: {endpoint_config_name}")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print(f"🚀 Creating fresh endpoint: {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
    print(f"✅ MLOps Deployment routine completed for: {endpoint_name}")

if __name__ == "__main__":
    deploy_or_update_endpoint()
