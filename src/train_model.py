import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(n_estimators=100, max_depth=None):
    """
    Train a RandomForest model on the Iris dataset
    with given parameters.
    Parameters:
    - n_estimators (int): Number of trees in the forest.
    - max_depth (int): Maximum depth of the tree. If None,
      nodes are expanded until all leaves are pure.

    Returns:
    - accuracy (float): Accuracy of the trained model.
    """
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model
    model_path = f"models/iris_model_{n_estimators}_{max_depth}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    return accuracy


if __name__ == "__main__":

    train_model()
