import os
import joblib
from src.train_model import train_model


def test_model_training():
    """
    Test if the train_model function successfully trains a model
    and saves it to the specified file.
    """
    # Define the directory where the model is saved
    model_dir = "models"

    # Ensure the models directory is clean
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, file))
    else:
        os.makedirs(model_dir)

    # Call the train_model function
    train_model()

    # Find the saved model file
    saved_model_file = next(
        (file for file in os.listdir(model_dir) if file.startswith("iris_model_")), None
    )

    # Assert that the model file was created
    assert saved_model_file is not None, "Model file was not created."
    model_path = os.path.join(model_dir, saved_model_file)

    # Load the saved model and verify it's not None
    model = joblib.load(model_path)
    assert model is not None, "Failed to load the saved model."


def test_model_accuracy_threshold():
    """
    Test if the trained model meets an acceptable accuracy threshold.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Call the train_model function
    train_model()

    # Find the saved model file
    model_dir = "models"
    saved_model_file = next(
        (file for file in os.listdir(model_dir) if file.startswith("iris_model_")), None
    )
    assert saved_model_file is not None, "Model file was not created."
    model_path = os.path.join(model_dir, saved_model_file)

    # Load the trained model
    model = joblib.load(model_path)

    # Predict and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Assert accuracy threshold
    assert accuracy >= 0.9, (
        f"Model accuracy {accuracy:.2f} is below acceptable threshold of 0.90."
    )
