import os
import json
import joblib
import numpy as np

# Load model
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

# Input processing
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

# Prediction
def predict_fn(input_data, model):
    return model.predict(input_data)

# Output formatting
def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
