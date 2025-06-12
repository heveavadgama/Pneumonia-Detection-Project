# 🩺 Pneumonia Detection from Chest X-rays (Normal | Bacterial | Viral)

This project leverages deep learning to classify chest X-ray images into **Normal**, **Bacterial Pneumonia**, or **Viral Pneumonia**. It is built using FastAI and ResNet50 to help automate medical image diagnosis and assist healthcare professionals in identifying the type of pneumonia a patient might have.

---

## 💡 Project Overview

- 🔍 **Goal:** Classify chest X-ray images into 3 classes: **NORMAL**, **BACTERIAL_PNEUMONIA**, and **VIRAL_PNEUMONIA**
- 🧠 **Model:** ResNet50 (transfer learning)
- 🎯 **Approach:**
  - Used FastAI's high-level APIs for data handling, augmentation, training, and visualization.
  - Combined and relabeled the dataset from Kaggle based on filenames (bacteria/virus).
  - Trained and evaluated the model on Google Colab using GPU.

---

## 🛠️ Tech Stack

- **Framework:** [FastAI](https://www.fast.ai/)  
- **Model Architecture:** ResNet50  
- **Language:** Python  
- **Platform:** Google Colab  
- **Dataset Source:** [Kaggle – Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Libraries Used:** `fastai`, `matplotlib`, `PIL`, `shutil`, `pathlib`, `os`

---

## 📊 Results

### ✅ Confusion Matrix – 3 Class Model (ResNet50)

| Actual \ Predicted     | BACTERIAL | NORMAL | VIRAL |
|------------------------|-----------|--------|-------|
| BACTERIAL_PNEUMONIA    | 463       | 6      | 66    |
| NORMAL                 | 5         | 320    | 12    |
| VIRAL_PNEUMONIA        | 97        | 8      | 194   |

- **Observation:**
  - The model performs well in identifying **Normal** and **Bacterial Pneumonia**.
  - There is some confusion between **Viral** and **Bacterial** classes, which is expected due to visual similarity in X-rays.
  - Normal class has very low false positives and false negatives.

---

## 🔍 Sample Prediction

On testing with a viral pneumonia image:

```bash
Predicted Class: VIRAL_PNEUMONIA

Class Probabilities:
  BACTERIAL_PNEUMONIA: 0.1248
  NORMAL: 0.0012
  VIRAL_PNEUMONIA: 0.8740

