# 🍎 FreshHarvest Classifier

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An AI-powered fruit freshness classifier using deep learning to determine if fruits are fresh or spoiled**

[Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Dataset](#-dataset) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🎬 Demo](#-demo)
- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [📖 Usage](#-usage)
- [🏗️ Model Architecture](#-model-architecture)
- [📊 Dataset](#-dataset)
- [🎓 Training](#-training)
- [📈 Performance](#-performance)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

FreshHarvest Classifier is an advanced machine learning system that uses **computer vision** to classify the **freshness** of fruits. The application is built using **PyTorch's ResNet-50** architecture and deployed as an **interactive Streamlit web app**, making it both robust and user-friendly.

This tool is designed to help individuals and businesses reduce food waste by offering accurate freshness classifications for fruits. Whether you're checking the condition of fruit at home or automating inspections in a supply chain, FreshHarvest Classifier is here to assist!

---

## ✨ Features

### 🤖 AI Capabilities
- **Transfer Learning**: Fine-tuned ResNet-50 pretrained on ImageNet for accurate representations.
- **Multi-class Classification**: Handles **16 classes** (8 fruits in Fresh and Spoiled categories).
- **Confidence Scoring**: Outputs the classification result with a detailed confidence level.
- **Data Augmentation**: Incorporates random flips, color jittering, and normalization for high-generalization across unseen images.

### 🖥️ Web Application
- **Drag-and-Drop**: Simplifies image upload with a straightforward interface.
- **Real-Time Results**: Instantly classify fruit freshness with predictions in seconds.
- **Interactive Visualizations**: Displays confidence percentages in a user-friendly bar chart.
- **Device Responsive**: Compatible across diverse screen sizes (desktop, tablets, mobile).
- **Professional UI**: Styled with custom CSS for modern aesthetics.

### 🛠️ Deployment Ready
- **Optimized for CPU**: Designed to run seamlessly on local devices without GPUs.
- **Extensible**: Easy-to-extend for additional categories or datasets.

---

## 🎬 Demo

### Supported Fruits

| Fresh Fruits        | Spoiled Fruits        |
|---------------------|-----------------------|
| 🍌 Banana           | 🍌 Banana (Spoiled)   |
| 🍋 Lemon            | 🍋 Lemon (Spoiled)    |
| 🟠 Lulo             | 🟠 Lulo (Spoiled)     |
| 🥭 Mango            | 🥭 Mango (Spoiled)    |
| 🍊 Orange           | 🍊 Orange (Spoiled)   |
| 🍓 Strawberry       | 🍓 Strawberry (Spoiled) |
| 🍅 Tamarillo        | 🍅 Tamarillo (Spoiled) |
| 🍅 Tomato           | 🍅 Tomato (Spoiled)   |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+ installed locally.

### Steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/freshharvest-classifier.git
   cd freshharvest-classifier
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. Download or prepare the **FRUIT-16K dataset** and place it like so:
   ```
   FreshHarvest/
   └── FRUIT-16K/
   ```

---

## ⚡ Quick Start

1. **Launch the Web Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   - Open your browser and navigate to `http://localhost:8501`.

3. **Drag and Drop a Fruit Image**:
   - Upload an image for classification.
   - Receive classification results and interactive visualizations.

---

## 📖 Usage

### API Usage

---

## 🏗️ Model Architecture

FreshHarvest uses a **fine-tuned ResNet-50** model:
- **Parameters**: ~23.5 million
- **Supported Input Size**: 128 × 128
- **Model Size**: ~94 MB
- **Optimized for**: High accuracy with efficient deployment.

---

## 📊 Dataset

### Structure
- **Images**: 16,000 total (70% Training, 15% Validation, 15% Test).
- **Categories**: Fresh (F_) and Spoiled (S_) fruits.

### Transformations Applied
- Random Resizing
- Normalization (ImageNet means/std).
- Data Augmentations (Color jittering, horizontal flips).

---

## 🎓 Training

1. Train the model:
   ```bash
   python train.py --epochs 10 --batch_size=64 --lr=1e-4
   ```

2. **Monitor Training**:
   Run TensorBoard during training:
   ```bash
   tensorboard --logdir=runs
   ```

3. Save the best model checkpoint:
   ```bash
   python evaluate.py --save best_model.pth
   ```

---

## 📈 Performance

| Metric            | Score    |
|--------------------|----------|
| **Validation Accuracy** | 92.7%   |
| **Test Accuracy**       | 91.3%   |
| **Inference Time (CPU)**| ~50 ms  |
| **Trainable Parameters**| ~23.5M  |

Performance benchmarks ensure **accuracy without GPU dependency**.

---

## 🤝 Contributing

We welcome contributions to enhance FreshHarvest!

### Steps to Contribute
1. Fork the `freshharvest-classifier` repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-xyz
   ```

3. Make improvements and push:
   ```bash
   git push origin feature-xyz
   ```

4. Submit a **Pull Request**!

---

## 📜 License

This project is licensed under the **MIT License**—feel free to fork, modify, and share!

---

## 🙏 Acknowledgments

### Special Thanks
- The **[PyTorch](https://pytorch.org/)** library for an excellent deep learning framework.
- **Streamlit** for enabling simple, interactive web app deployments.

### Dataset Contributors
- Thanks to creators of FRUIT-16K for providing labeled images for both fresh and spoiled fruits.

---

<div align="center">

**⭐ If you liked this project, give it a star!**

[Report Bug](#) • [Request Feature](#)

Built with 🚀 by the FreshHarvest Team

</div>