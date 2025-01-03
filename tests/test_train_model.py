import os
import joblib
from src.train_model import train_model

def test_model_training():
    """
    Test if the train_model function successfully trains a model
    and saves it to the specified file.
    """
    # Ensure no residual model file exists
    model_path = "models/iris_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Call the train_model function
    train_model()

    # Assert that the model file was created
    assert os.path.exists(model_path), "Model file was not created."

    # Load the saved model and verify it's not None
    model = joblib.load(model_path)
    assert model is not None, "Failed to load the saved model."

def test_model_accuracy_threshold():
    """
    Test if the trained model meets an acceptable accuracy threshold.
    """
    # Call the train_model function
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the train_model function
    train_model()

    # Load the trained model
    model_path = "models/iris_model.pkl"
    model = joblib.load(model_path)

    # Predict and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Assert accuracy threshold
    assert accuracy >= 0.9, f"Model accuracy {accuracy:.2f} is below acceptable threshold of 0.90."
