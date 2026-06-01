from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

try:
    from .preprocess import (
        IMAGE_SIZE,
        NUM_CLASSES,
        preprocess_emnist_letter_example,
        preprocess_mnist_example,
    )
except ImportError:
    from preprocess import (
        IMAGE_SIZE,
        NUM_CLASSES,
        preprocess_emnist_letter_example,
        preprocess_mnist_example,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "handwritten_characters_cnn.keras"
AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(
    name: str,
    split: str,
    data_dir: Path,
) -> tf.data.Dataset:
    dataset = tfds.load(
        name,
        split=split,
        as_supervised=True,
        data_dir=str(data_dir),
    )
    preprocess = (
        preprocess_emnist_letter_example
        if name == "emnist/letters"
        else preprocess_mnist_example
    )
    return dataset.map(preprocess, num_parallel_calls=AUTOTUNE)


def build_dataset(
    mnist_split: str,
    emnist_split: str,
    data_dir: Path,
    batch_size: int,
    training: bool = False,
) -> tf.data.Dataset:
    mnist = load_dataset("mnist", mnist_split, data_dir)
    letters = load_dataset("emnist/letters", emnist_split, data_dir)
    dataset = mnist.concatenate(letters).cache()
    if training:
        dataset = dataset.shuffle(20_000, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = layers.RandomTranslation(0.08, 0.08)(inputs)
    x = layers.RandomRotation(0.06)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="handwritten_characters_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST and EMNIST Letters. The first run downloads them automatically.")
    train_dataset = build_dataset(
        "train[:90%]", "train[:90%]", args.data_dir, args.batch_size, training=True
    )
    validation_dataset = build_dataset(
        "train[90%:]", "train[90%:]", args.data_dir, args.batch_size
    )
    test_dataset = build_dataset("test", "test", args.data_dir, args.batch_size)

    model = build_model()
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    best_model = keras.models.load_model(args.model_path)
    test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Saved best model to: {args.model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
