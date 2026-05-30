import os
import shutil
import argparse
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    print("Starting training execution loop inside production workspace...")
    
    # Load standardized feature selection matrix matching test structures
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Simple algorithm configuration fitting signature shapes
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Training cycle completed successfully. Evaluation score value: {model.score(X_test, y_test):.4f}")
    
    # Core Cloud Isolation Fix: Define standard structural output directory target
    model_dir = os.environ.get("SAGEMAKER_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Store model weights object directly inside the base archive path
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model weight binaries cleanly persisted natively: {model_path}")
    
    # 2. Package the execution script directly into the artifact archive 🎯
    container_code_dir = os.path.join(model_dir, "code")
    os.makedirs(container_code_dir, exist_ok=True)
    
    # Search for execution scripts dynamically based on local container runtime contexts
    possible_source_paths = [
        "inference.py",
        os.path.join(os.getcwd(), "inference.py"),
        os.path.join(os.path.dirname(__file__), "inference.py")
    ]
    
    script_copied = False
    for path in possible_source_paths:
        if os.path.exists(path):
            shutil.copy(path, os.path.join(container_code_dir, "inference.py"))
            print(f"✅ Successfully nested code module from path target source: {path}")
            script_copied = True
            break
            
    if not script_copied:
        print("⚠️ Warning: inference.py script was not found dynamically during the asset compilation phase.")

if __name__ == "__main__":
    train()
