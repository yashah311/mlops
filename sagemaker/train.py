import os
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# SageMaker specific dirs
model_dir = os.environ.get("SM_MODEL_DIR")

# Load dataset
data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save model
path = os.path.join(model_dir, "model.joblib")
joblib.dump(model, path)

print("Model trained and saved at:", path)
