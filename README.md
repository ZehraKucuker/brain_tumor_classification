# 🧠 Brain Tumor Detection and Classification

Vision Transformer (ViT) based deep learning project for brain tumor detection and classification from MRI images.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.87%25-green.svg)

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#️-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)

## 🎯 About the Project

This project is developed for tumor detection and classification from MRI brain images. Using Vision Transformer (ViT) model, it performs high-accuracy classification between 4 different classes:

- **Glioma** - Tumor originating from glial cells
- **Meningioma** - Tumor originating from meninges membrane  
- **Pituitary** - Pituitary gland tumor
- **Healthy** - Healthy brain

## ✨ Features

- 🔬 **Image Preprocessing Pipeline**
  - Automatic black border cropping
  - Noise reduction with bilateral filter
  - CLAHE contrast enhancement
  - Standard resizing (224x224)

- 🤖 **Vision Transformer Model**
  - Pre-trained ViT-Small model
  - Transfer learning
  - Data augmentation

- 📊 **Comprehensive Evaluation**
  - Classification metrics (Precision, Recall, F1-Score)
  - Confusion matrix
  - ROC curve and AUC scores

## 📁 Dataset

### 📥 Download Dataset

The raw (unprocessed) dataset used in this project is shared on Kaggle. You can easily download it from the link below:

<p align="center">
  <a href="https://www.kaggle.com/datasets/zehrakucuker/brain-tumor-mri-images-classification-dataset" target="_blank">
    <img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Dataset"/>
  </a>
</p>

🔗 **[Brain Tumor MRI Images Classification Dataset](https://www.kaggle.com/datasets/zehrakucuker/brain-tumor-mri-images-classification-dataset)**

> 💡 **Note:** This dataset is created by combining 3 different Kaggle datasets.

### 📊 Dataset Statistics

| Class | Image Count | Ratio |
|-------|-------------|-------|
| Glioma | 3,768 | 24.1% |
| Healthy | 3,990 | 25.6% |
| Meningioma | 3,806 | 24.4% |
| Pituitary | 4,041 | 25.9% |
| **Total** | **15,605** | 100% |

## 🚀 Installation

### Requirements

- Python 3.10+
- CUDA enabled GPU (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ZehraKucuker/brain_tumor_classification.git
cd brain_tumor_classification
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (for GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm
```

## 💻 Usage

### Jupyter Notebook

All project code is consolidated into a single interactive notebook file:

```bash
jupyter notebook brain_tumor_classification.ipynb
```

Or you can open it directly in VS Code.

### Notebook Content

**1. Dataset Analysis**
- Number of images in each class
- Image dimensions and statistics
- Pixel value distributions
- Class distribution charts

**2. Image Preprocessing**
- Black border cropping
- Bilateral filter (noise reduction)
- CLAHE (contrast enhancement)
- 224x224 resizing

**3. Model Training**
- Vision Transformer (ViT-Small) model
- Transfer learning training
- Data augmentation

**4. Model Evaluation**
- Confusion Matrix
- ROC curves
- Classification report

### Configuration

Parameters can be adjusted from the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 1e-4,
    'model_name': 'vit_small_patch16_224',
    ...
}
```

## 🏗️ Model Architecture

```
Vision Transformer (ViT-Small)
├── Patch Embedding (16x16 patches)
├── Transformer Encoder (12 layers)
│   ├── Multi-Head Self-Attention
│   └── MLP Block
├── Classification Head
└── Output: 4 classes
```

**Model Features:**
- Total Parameters: 21,667,204
- Patch Size: 16x16
- Input Size: 224x224x3
- Pre-trained: ImageNet-21k

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.87%** |
| **Macro F1-Score** | 0.9987 |
| **Weighted F1-Score** | 0.9987 |

### Class-wise Results

| Class | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Glioma | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Meningioma | 0.9947 | 1.0000 | 0.9973 | 1.0000 |
| Pituitary | 1.0000 | 0.9949 | 0.9975 | 1.0000 |

### Visualizations

All graphs (training history, confusion matrix, ROC curves) are displayed inline within the notebook.

## 📂 Project Structure

```
brain_tumor_classification/
│
├── dataset/                          # Original dataset
│   ├── glioma/
│   ├── healthy/
│   ├── meningioma/
│   └── pituitary/
│
├── dataset_processed/                # Processed dataset (224x224)
│   ├── glioma/
│   ├── healthy/
│   ├── meningioma/
│   └── pituitary/
│
├── .venv/                            # Python virtual environment
│
├── brain_tumor_classification.ipynb  # Main notebook (all code)
├── requirements.txt                  # Python dependencies
├── best_model.pth                    # Trained model weights
│
└── README.md                         # This file
```

## 🔧 Configuration

Main configuration parameters in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Mini-batch size |
| `epochs` | 15 | Number of training epochs |
| `learning_rate` | 1e-4 | Learning rate |
| `image_size` | 224 | Input image size |
| `model_name` | vit_small_patch16_224 | ViT model variant |
| `train_split` | 0.8 | Training set ratio |
| `val_split` | 0.1 | Validation set ratio |
| `test_split` | 0.1 | Test set ratio |

## 📚 Dependencies

```
numpy
pandas
matplotlib
seaborn
opencv-python
scikit-learn
scikit-image
Pillow
tqdm
torch
torchvision
timm
```
