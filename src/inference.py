import os
import json
import joblib
import numpy as np

def model_fn(model_dir):
    """Loads the model artifact from the bundle directory when the container initializes."""
    print("Executing custom model_fn initialization stage...")
    model_file = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Missing required model artifact at path: {model_file}")
    model = joblib.load(model_file)
    print("Successfully loaded model artifact file into framework memory runtime.")
    return model

def input_fn(request_body, request_content_type):
    """Parses incoming deployment traffic payloads into array shapes."""
    print(f"Received request payload content type: {request_content_type}")
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported input content formatting protocol: {request_content_type}")

def predict_fn(input_data, model):
    """Handles runtime inference calculations against the active operational model."""
    print(f"Processing prediction against matrix shape: {input_data.shape}")
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """Formats inference calculations into valid response text streams.
    
    CRITICAL FIX: Prevents the container from crashing with a 500 /ping error.
    """
    print(f"Formatting outbound response content protocol: {response_content_type}")
    
    # Convert numpy array to standard Python list for JSON serialization
    if isinstance(prediction, np.ndarray):
        response_body = json.dumps(prediction.tolist())
    else:
        response_body = json.dumps(prediction)
        
    return response_body, "application/json"
