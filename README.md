# 😷 Real-Time Face Mask Detection using PyTorch

This project implements a **real-time face mask detection system** using **PyTorch** and **OpenCV**.  
It detects whether a person is wearing a mask or not from live webcam feed — ideal for public safety monitoring, healthcare, and smart surveillance systems.

---

## 🚀 Features

- Real-time mask detection using webcam 📷  
- Deep learning model built with **MobileNetV2 (PyTorch)**  
- Trained on **Kaggle Face Mask Dataset**  
- Visualization of training accuracy, confusion matrix, ROC curve, and F1 metrics  
- Lightweight and fast — runs on CPU/GPU (Apple M2, CUDA, etc.)

---

## 🧠 Dataset

Dataset used:  
📦 **[Face Mask Dataset – Omkar Gurav (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)**

- `with_mask` — images of people wearing masks  
- `without_mask` — images of people without masks  

After downloading, place the dataset in:

dataset/
├── with_mask/
└── without_mask/


The preprocessing script will automatically split into train, validation, and test sets.

---

## 🧩 Project Structure

mask-detection-pytorch/
│
├── data/
│ └── split/
│ ├── train/
│ ├── val/
│ └── test/
│
├── models/
│ ├── mask_detector.pth
│ └── training_history.pth
│
├── results/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── precision_recall_curve.png
│ ├── metric_summary.png
│ ├── accuracy_curve.png
│ └── loss_curve.png
│
├── preprocess.py
├── train.py
├── evaluate.py
├── realtime.py
├── evaluation_graphs.py
├── requirements.txt
└── README.md




