import mlflow
import mlflow.sklearn
from train_model import train_model

# Define parameter combinations for experiments
params = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 4},
]

# Run experiments with MLflow tracking
if __name__ == "__main__":
    mlflow.set_experiment("Iris_Model_Experiments")

    for param in params:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", param["n_estimators"])
            mlflow.log_param("max_depth", param["max_depth"])

            # Train the model and log the accuracy
            accuracy = train_model(param["n_estimators"], param["max_depth"])
            mlflow.log_metric("accuracy", accuracy)

            print(f"Experiment with params {param} completed."
                  "Accuracy: {accuracy:.2f}")
