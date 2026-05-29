import os
import shutil
import argparse
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    print("Starting training inside SageMaker container...")
    
    # Load sample diagnostic dataset matching your payload shapes
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Simple model configuration
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained successfully. Accuracy score: {model.score(X_test, y_test):.4f}")
    
    # Core Fix: Use the canonical SageMaker output artifact path location
    model_dir = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the binary object
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model artifact saved natively to: {model_path}")
    
    # Enforce container bundle rule: Clone serving files into model archive root
    container_code_dir = os.path.join(model_dir, "code")
    os.makedirs(container_code_dir, exist_ok=True)
    
    # Copy inference definition file from current folder into output payload directory
    if os.path.exists("inference.py"):
        shutil.copy("inference.py", os.path.join(container_code_dir, "inference.py"))
        print("Embedded inference.py inside model code container path.")
    else:
        # Fallback if execution path differs slightly during tracking
        src_inference = os.path.join(os.path.dirname(__file__), "inference.py")
        if os.path.exists(src_inference):
            shutil.copy(src_inference, os.path.join(container_code_dir, "inference.py"))
            print("Embedded inference.py from source absolute directory.")

if __name__ == "__main__":
    train()

