from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

try:
    from .preprocess import preprocess_uploaded_image
except ImportError:
    from preprocess import preprocess_uploaded_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "handwritten_characters_cnn.keras"
CHARACTERS = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def predict_character(
    image_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH
) -> tuple[str, float]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python src/train.py"
        )

    model = keras.models.load_model(model_path)
    probabilities = model.predict(preprocess_uploaded_image(image_path), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return CHARACTERS[predicted_index], float(probabilities[predicted_index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    character, confidence = predict_character(args.image, args.model_path)
    print(f"Prediction: {character}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
