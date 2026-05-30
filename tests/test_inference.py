import os
import sys
import yaml
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

def run_integration_test():
    print("🧪 Starting parameterized Endpoint Integration Test Gate...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    sagemaker_client = boto3.client("sagemaker", region_name=config["aws"]["region"])
    
    # 🎯 FIX: Draw parameter dynamically from configuration layout mapping
    endpoint_name = config["aws"]["endpoint_name"]
    print(f"Targeting deployment endpoint name parameter: {endpoint_name}")
    
    try:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print(f"Target Endpoint '{endpoint_name}' located with status context: {status}")
        
        if status != "InService" and status != "Updating":
            print(f"❌ Test Failed: Endpoint infrastructure is not ready (Status: {status}).")
            sys.exit(1)
            
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        sample_payload = [[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
            0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654,
            0.4601, 0.1189
        ]]
        
        print("Sending verification tracking matrix payload to endpoint...")
        result = predictor.predict(sample_payload)
        print(f"Live Prediction Response Matrix Received: {result}")
        
        if result is not None:
            print("✅ Success: Endpoint is live and evaluating signatures perfectly. Conclusion: 0")
            sys.exit(0)
        else:
            print("❌ Test Failed: Empty prediction response matrix received.")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Critical Integration Failure encountered: {str(e)}")
        print("Conclusion: 1")
        sys.exit(1)

if __name__ == "__main__":
    run_integration_test()
