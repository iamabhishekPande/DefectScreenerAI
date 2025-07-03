# 🔍 Defect Detection Using ResNet-50

The objective of this project is to develop a deep learning model that can automatically classify images of screws as either defective or non-defective. This model is designed to assist in automated visual inspection by identifying screws that exhibit issues such as scratches, missing threads, surface damage, or any other form of visible manipulation. By fine-tuning a ResNet-50 architecture, the model learns to distinguish between high-quality screws and those with physical defects, aiming to enhance the speed, consistency, and accuracy of quality control in industrial environments.

---
## 📁 Project Overview

├── resnet50.py # Training script
├── eval.py # Evaluation script
├── predict.py # Prediction script
├── model/
│ ├── resnet50_final.pt # Final trained model
│ ├── confusion_matrix.png # Confusion matrix visualization
│ ├── false_positive.png # Examples of false positives
│ ├── false_negative.png # Examples of false negatives
│ ├── metrics_report.txt # Evaluation metrics summary
│ └── prediction.txt # Final predictions on unseen data
├── finalTestImages/ # Folder containing test images for prediction


---

## 🖼️ Dataset Preparation

The dataset was manually collected and categorized into two classes: **defective** and **non-defective**. 

### Steps followed for dataset preparation:

- All images were resized to **512×512 pixels** for uniformity.
- The dataset was split into:
  - **70%** for training
  - **20%** for validation
  - **10%** for testing
- The folder structure follows the expected format for PyTorch’s.  
finaldataset/
├── train/
├── valid/
└── test/
- Data augmentation was applied **only to the training set** to improve generalization.

---

## 🧪 Data Augmentation

The following augmentations were applied to the training images:

- Horizontal Flip
- Vertical Flip
- Random Rotation (±15°)
- Grayscale conversion (10% of images)
- Hue adjustment (±12°)
- Brightness adjustment (±10%)
- Exposure adjustment (±10%)

These were implemented using `torchvision.transforms`.

---

## 🧠 Model Architecture

- Base model: **ResNet-50**, pretrained on ImageNet
- Final fully connected layer replaced to support **binary classification**
- Input image size: **512×512 RGB**

---

## 🏋️‍♂️ Training Strategy

The model was trained in **two phases** using the `resnet50.py` script:

### Phase 1: Feature Extraction
- 80% of the layers were **frozen**
- Trained for **10 epochs**
- Optimizer: Adam with learning rate `1e-4`

### Phase 2: Fine-Tuning
- All layers were **unfrozen**
- Trained for an additional **40 epochs**
- Optimizer: Adam with learning rate `1e-5`
- Learning rate scheduler: StepLR (step size=10, gamma=0.5)

The final model is saved as `resnet50_final.pt` in the `model/` folder.

---

## 📊 Model Performance

The model achieved strong results on the validation and test sets:

| Metric         | Value           |
|----------------|-----------------|
| Accuracy       | 0.9500          |
| Precision      | 0.9394          |
| Recall         | 1.0000          |
| F1 Score       | 0.9688          |
| Inference Time | 0.001899 sec/image (CPU) |
| Model Params   | 23,512,130      |
| Model File Size| 94.37 MB        |

A full breakdown of these metrics can be found in:  
📄 `model/metrics_report.txt`

---

## 📈 Evaluation Results

- **Confusion Matrix** → `model/confusion_matrix.png`
- **False Positives** → `model/false_positive.png`
- **False Negatives** → `model/false_negative.png`

These files help visualize how the model is performing and where it's making mistakes.

---

## 🔍 Final Predictions

After training and evaluation, the final model was tested on a separate set of images stored in `finalTestImages/`.

- Predictions were made using `predict.py`
- Output results are stored in `model/prediction.txt`

---

## 💡 How to Use

1. Train the Model

    python resnet50.py

2. Evaluate the model 

    python eval.py

3. Run Prediction 

    python predict.py


⚙️ Dependencies

Key packages:

Python 3.8+

PyTorch

Torchvision

NumPy

Matplotlib

tqdm

skikit-learn



