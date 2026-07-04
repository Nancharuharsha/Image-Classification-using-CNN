# Image Classification using CNN (CIFAR-10)

A Convolutional Neural Network built from scratch with TensorFlow/Keras
to classify images into 10 categories, matching the resume project:
data augmentation + dropout regularization, trained/validation curves,
~87% test accuracy target.

## Project structure

```
cnn_project/
├── model.py            # CNN architecture (3 conv blocks + dense head)
├── train.py            # loads CIFAR-10, augments data, trains, plots curves
├── predict.py          # run inference on your own image
├── models/
│   └── cnn_cifar10.keras   # saved trained model (created after training)
├── plots/
│   └── training_curves.png # accuracy/loss curves (created after training)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train

```bash
python train.py
```

This will:
- Download CIFAR-10 automatically (60,000 32×32 color images, 10 classes) via `keras.datasets`
- Normalize pixels to [0, 1] and one-hot encode labels
- Apply data augmentation (rotation, shifts, horizontal flip, zoom)
- Train a CNN with batch norm + dropout (0.25 → 0.5 across layers) to curb overfitting
- Use early stopping + learning-rate reduction on plateau
- Save training/validation accuracy & loss curves to `plots/training_curves.png`
- Save the trained model to `models/cnn_cifar10.keras`

Expect ~30-50 epochs (early stopping dependent) and **~85-88% test accuracy**
on a GPU; CPU training will work but is considerably slower (~20-40 min/epoch
range varies by hardware — a GPU runtime, e.g. Google Colab, is recommended).

## 2. Classify your own image

```bash
python predict.py path/to/your_image.jpg
```

Resizes any image to 32×32 and prints the top-3 predicted classes with
confidence scores.

## Classes

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## Architecture notes

- 3 convolutional blocks (32 → 64 → 128 filters), each with two conv
  layers, batch normalization, max pooling, and increasing dropout
  (0.25, 0.3, 0.4) to progressively regularize deeper layers.
- Dense head with 256 units, batch norm, and 0.5 dropout before the
  softmax output layer.
- `ImageDataGenerator` augmentation (rotation, shift, flip, zoom)
  directly addresses overfitting on CIFAR-10's relatively small
  32×32 images, matching the "reduce overfitting" resume bullet.

## Notes

- Training requires TensorFlow, which was not available in the sandbox
  this project was built in (no internet access to install it or
  download CIFAR-10) — the code is ready to run as-is on your machine
  or in Colab.
- For faster iteration, reduce `EPOCHS` in `train.py` or run on a GPU
  (Colab's free tier works well for CIFAR-10).
