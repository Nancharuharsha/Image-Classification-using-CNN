"""
train.py
--------
Trains the CNN defined in model.py on CIFAR-10.

- Loads CIFAR-10 (60,000 32x32 color images, 10 classes)
- Normalizes pixel values and one-hot encodes labels
- Applies data augmentation (rotation, shift, flip, zoom) to reduce overfitting
- Trains with dropout + batch norm regularization
- Plots and saves training/validation accuracy & loss curves
- Saves the trained model to models/cnn_cifar10.keras
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import build_cnn

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

EPOCHS = 50
BATCH_SIZE = 64


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train_cat), (x_test, y_test_cat)


def get_augmenter():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )


def plot_history(history, out_path="plots/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    print("Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    augmenter = get_augmenter()
    augmenter.fit(x_train)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    plot_history(history)

    model.save("models/cnn_cifar10.keras")
    print("Saved trained model to models/cnn_cifar10.keras")


if __name__ == "__main__":
    main()
