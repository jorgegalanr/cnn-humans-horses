# cnn-humans-horses (VGG16 / ResNet50)

📌 Description

Image classification project using Convolutional Neural Networks with Transfer Learning.
We fine-tune a pretrained backbone (VGG16 / ResNet50) on the Humans vs Horses dataset, apply data augmentation, and compare two strategies: feature extraction vs fine-tuning.

🎯 Objectives

- Build a reproducible deep-learning pipeline.

- Compare feature extractor (frozen backbone) vs fine-tuning (unfreezing top blocks).

- Use data augmentation to improve generalization.

- Report accuracy, macro-F1, training curves, and a confusion matrix.

📊 Dataset

Two classes in folders:

- human

- horse
  
⚙️ Tech Stack

- Python

- TensorFlow/Keras (transfer learning, data augmentation)

- scikit-learn (metrics/confusion matrix)

- Matplotlib/Seaborn (plots)

- TensorBoard (optional)

🔎 Methodology

1. Data Preparation

-  Directory loaders (train/val/test) with augmentation: random flips, rotation, zoom/shift.

2. Modeling

- Backbone: VGG16 / ResNet50 (imagenet, include_top=False).

- Head: GAP → Dense (Dropout opcional) → Sigmoid.

- Two phases:

    * Feature extractor (freeze backbone).

    * Fine-tuning (unfreeze top blocks with small LR).

3. Training & Evaluation

- Optimizer: Adam; callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

- Metrics: accuracy, macro-precision/recall/F1 + confusion matrix.

4. Visualization

- Learning curves (loss/accuracy).

- Confusion matrix en reports/figures/.

📈 Results

Expected after fine-tuning: high accuracy and clean separation between classes.
Add your figures after training:

- Training curves

- Confusion matrix


⚙️ requirements.txt

tensorflow==2.16.1
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
pandas==2.2.2
numpy==1.26.4


