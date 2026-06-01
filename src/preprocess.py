from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMAGE_SIZE = 28
NUM_CLASSES = 36


def preprocess_mnist_example(
    image: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.int32)


def preprocess_emnist_letter_example(
    image: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    # TFDS stores EMNIST images transposed.
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32) + 9
    tf.debugging.assert_less(label, NUM_CLASSES)
    return image, label


def preprocess_uploaded_image(image_source: str | Path | Image.Image) -> np.ndarray:
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        image = Image.open(image_source)

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    if np.asarray(image, dtype=np.float32).mean() > 127:
        image = ImageOps.invert(image)

    bounding_box = image.getbbox()
    if bounding_box:
        image = image.crop(bounding_box)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    left = (IMAGE_SIZE - image.width) // 2
    top = (IMAGE_SIZE - image.height) // 2
    canvas.paste(image, (left, top))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
