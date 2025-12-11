# 🧠 Beyin Tümörü Tespiti ve Sınıflandırması

MRI görüntülerinden beyin tümörü tespiti ve sınıflandırması için Vision Transformer (ViT) tabanlı derin öğrenme projesi.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.87%25-green.svg)

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Veriseti](#-veriseti)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarisi](#️-model-mimarisi)
- [Sonuçlar](#-sonuçlar)
- [Proje Yapısı](#-proje-yapısı)

## 🎯 Proje Hakkında

Bu proje, MRI beyin görüntülerinden tümör tespiti ve sınıflandırması yapmak için geliştirilmiştir. Vision Transformer (ViT) modeli kullanılarak 4 farklı sınıf arasında yüksek doğrulukla sınıflandırma yapılmaktadır:

- **Glioma** - Glial hücrelerden kaynaklanan tümör
- **Meningioma** - Meninks zarından kaynaklanan tümör  
- **Pituitary** - Hipofiz bezi tümörü
- **Healthy** - Sağlıklı beyin

## ✨ Özellikler

- 🔬 **Görüntü Ön İşleme Pipeline'ı**
  - Otomatik siyah kenar kırpma
  - Bilateral filtre ile gürültü giderme
  - CLAHE kontrast artırma
  - Standart boyutlandırma (224x224)

- 🤖 **Vision Transformer Modeli**
  - Pre-trained ViT-Small modeli
  - Transfer learning
  - Data augmentation

- 📊 **Kapsamlı Değerlendirme**
  - Sınıflandırma metrikleri (Precision, Recall, F1-Score)
  - Karmaşıklık matrisi
  - ROC eğrisi ve AUC skorları

## 📁 Veriseti

| Sınıf | Görüntü Sayısı | Oran |
|-------|----------------|------|
| Glioma | 3,768 | %24.1 |
| Healthy | 3,990 | %25.6 |
| Meningioma | 3,806 | %24.4 |
| Pituitary | 4,041 | %25.9 |
| **Toplam** | **15,605** | %100 |

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilir)

### Adım 1: Repoyu Klonlayın

```bash
git clone https://github.com/ZehraKucuker/brain_tumor_classification.git
cd brain_tumor_classification
```

### Adım 2: Sanal Ortam Oluşturun

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: PyTorch'u CUDA ile Yükleyin (GPU için)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm
```

## 💻 Kullanım

*Güncellencek*

Eğitim parametreleri `train_vit.py` içindeki `CONFIG` sözlüğünden ayarlanabilir:

```python
CONFIG = {
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 1e-4,
    'model_name': 'vit_small_patch16_224',
    ...
}
```

## 🏗️ Model Mimarisi

```
Vision Transformer (ViT-Small)
├── Patch Embedding (16x16 patches)
├── Transformer Encoder (12 layers)
│   ├── Multi-Head Self-Attention
│   └── MLP Block
├── Classification Head
└── Output: 4 sınıf
```

**Model Özellikleri:**
- Toplam Parametre: 21,667,204
- Patch Boyutu: 16x16
- Giriş Boyutu: 224x224x3
- Pre-trained: ImageNet-21k

## 📈 Sonuçlar

### Model Performansı

| Metrik | Değer |
|--------|-------|
| **Test Accuracy** | **99.87%** |
| **Macro F1-Score** | 0.9987 |
| **Weighted F1-Score** | 0.9987 |

### Sınıf Bazlı Sonuçlar

| Sınıf | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Glioma | 1.0000 | 0.9975 | 0.9987 | 1.0000 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Meningioma | 0.9947 | 1.0000 | 0.9973 | 1.0000 |
| Pituitary | 1.0000 | 0.9975 | 0.9987 | 1.0000 |

### Eğitim Grafikleri

![Eğitim Geçmişi](training_history.png)

### Karmaşıklık Matrisi

![Confusion Matrix](confusion_matrix.png)

### ROC Eğrisi

![ROC Curve](roc_curve.png)

## 📂 Proje Yapısı

```
brain_tumor_classification/
│
├── dataset/                    # Orijinal veriseti
│   ├── glioma/
│   ├── healthy/
│   ├── meningioma/
│   └── pituitary/
│
├── dataset_processed/          # İşlenmiş veriseti
│   ├── glioma/
│   ├── healthy/
│   ├── meningioma/
│   └── pituitary/
│
├── .venv/                      # Python sanal ortamı
│
├── dataset_analysis.py         # Veriseti analiz scripti
├── preprocessing.py            # Görüntü ön işleme scripti
├── train_vit.py                # Model eğitim scripti
├── requirements.txt            # Python bağımlılıkları
├── best_model.pth              # Eğitilmiş model ağırlıkları
│
├── dataset_analysis.png        # Veriseti analiz grafikleri
├── training_history.png        # Eğitim geçmişi grafikleri
├── confusion_matrix.png        # Karmaşıklık matrisi
├── confusion_matrix_normalized.png
├── roc_curve.png               # ROC eğrisi
│
└── README.md                   # Bu dosya
```

## 🔧 Konfigürasyon

`train_vit.py` içindeki ana konfigürasyon parametreleri:

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `batch_size` | 32 | Mini-batch boyutu |
| `epochs` | 15 | Eğitim epoch sayısı |
| `learning_rate` | 1e-4 | Öğrenme oranı |
| `image_size` | 224 | Giriş görüntü boyutu |
| `model_name` | vit_small_patch16_224 | ViT model varyantı |
| `train_split` | 0.8 | Eğitim seti oranı |
| `val_split` | 0.1 | Doğrulama seti oranı |
| `test_split` | 0.1 | Test seti oranı |

## 📚 Bağımlılıklar

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
