from PIL import Image
import numpy as np


def preprocess_mnist(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("L")

    image = image.resize((28, 28))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = image.reshape(1, 28, 28, 1)

    return image


def preprocess_emnist(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("L")

    image = image.resize((28, 28))

    image = np.array(image)

    # Match EMNIST orientation used during training
    image = np.transpose(image)

    image = np.fliplr(image)

    image = image.astype("float32") / 255.0

    image = image.reshape(1, 28, 28, 1)

    return image


def preprocess_image(uploaded_file, model_type):

    if model_type == "MNIST":
        return preprocess_mnist(uploaded_file)

    elif model_type == "EMNIST":
        return preprocess_emnist(uploaded_file)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
