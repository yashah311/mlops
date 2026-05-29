import os
import yaml
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

def test_endpoint_prediction():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    sagemaker_client = boto3.client("sagemaker", region_name=config["aws"]["region"])
    
    # Fetch active endpoint tracking entities matching our registry metadata signature
    # In production, this can also read a dynamic value stored in AWS Systems Manager Parameter Store
    response = sagemaker_client.list_endpoints(
        SortBy="CreationTime",
        SortOrder="Descending",
        NameContains="model-group" # Filters based on the model package group name structure
    )
    
    endpoints = response.get("Endpoints", [])
    if not endpoints:
        raise FileNotFoundError("No active target inference endpoints located for test verification.")
        
    active_endpoint_name = endpoints[0]["EndpointName"]
    print(f"Connecting evaluation harness to runtime endpoint target: {active_endpoint_name}")
    
    predictor = Predictor(
        endpoint_name=active_endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    # Standard 30-feature shape data mapping to Scikit-Learn breast cancer matrices
    sample_payload = [[
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
        0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654,
        0.4601, 0.1189
    ]]
    
    result = predictor.predict(sample_payload)
    print(f"Evaluation transaction completed successfully. Response output matrix: {result}")
    assert result is not None, "Endpoint returned an empty inference result payload."
    print("✅ Integration verification tests completed successfully.")

if __name__ == "__main__":
    test_endpoint_prediction()
