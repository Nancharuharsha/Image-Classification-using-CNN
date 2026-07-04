"""
predict.py
----------
Loads the trained CNN and classifies a single image (any size — it will
be resized to 32x32 to match CIFAR-10 input).

Usage:
    python predict.py path/to/image.jpg
"""

import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def predict(image_path, model_path="models/cnn_cifar10.keras"):
    model = load_model(model_path)

    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(probs)

    print(f"\nPrediction: {CLASS_NAMES[top_idx]} ({probs[top_idx]:.1%} confidence)")
    print("\nTop 3 predictions:")
    top3 = np.argsort(probs)[::-1][:3]
    for i in top3:
        print(f"  {CLASS_NAMES[i]:<12} {probs[i]:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
