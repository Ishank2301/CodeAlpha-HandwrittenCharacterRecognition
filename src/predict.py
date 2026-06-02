from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_image

BASE_DIR = Path(__file__).resolve().parent.parent

MNIST_MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"
EMNIST_MODEL_PATH = BASE_DIR / "models" / "emnist_cnn.keras"

EMNIST_LABELS = [chr(i) for i in range(65, 91)]

_model_cache = {}


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

print("BASE_DIR =", BASE_DIR)

print("Models folder exists:", (BASE_DIR / "models").exists())

print("EMNIST path:", BASE_DIR / "models" / "emnist_cnn.keras")

print("EMNIST exists:", (BASE_DIR / "models" / "emnist_cnn.keras").exists())


def get_model(model_path):
    model_path = str(model_path)

    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path)

    return _model_cache[model_path]


def predict(uploaded_file, model_type):

    image = preprocess_image(uploaded_file, model_type)

    if model_type == "MNIST":

        if not MNIST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MNIST_MODEL_PATH}")

        model = get_model(MNIST_MODEL_PATH)

        predictions = model.predict(image, verbose=0)

        predicted_class = int(np.argmax(predictions))

        confidence = float(np.max(predictions))

        return str(predicted_class), confidence

    elif model_type == "EMNIST":

        if not EMNIST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {EMNIST_MODEL_PATH}")

        model = get_model(EMNIST_MODEL_PATH)

        predictions = model.predict(image, verbose=0)

        predicted_index = int(np.argmax(predictions))

        confidence = float(np.max(predictions))

        predicted_character = EMNIST_LABELS[predicted_index]

        return predicted_character, confidence

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
