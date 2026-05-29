import os
import yaml
import boto3
import datetime

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
    
    # 2. Dynamic Naming Definitions
    endpoint_name = f"{config['aws']['model_package_group_name']}-prod"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    model_name = f"{config['aws']['model_package_group_name']}-model-{timestamp}"
    
    print(f"Creating Model Resource: {model_name}")
    print(f"Creating Unique Config: {endpoint_config_name}")
    print(f"Targeting Endpoint: {endpoint_name}")
    
    # 3. Create SageMaker Model Entity from the Model Package
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "ModelPackageName": latest_package_arn
        }
    )
    
    # 4. Create an Immutable, Timestamped Endpoint Configuration 🚀
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
    
    # 5. Check if Endpoint Already Exists to Determine Routing Strategy
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Active endpoint detected! Performing zero-downtime rolling update...")
        
        # Safely switch traffic to our unique timestamp configuration
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Update command sent to AWS successfully.")
        
    except sagemaker_client.exceptions.ClientError:
        print(f"Endpoint not found. Initializing a fresh deployment...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Fresh creation command sent to AWS successfully.")
        
    print(f"🎯 Production Endpoint deployment tracking active: {endpoint_name}")

if __name__ == "__main__":
    deploy_or_update_endpoint()
