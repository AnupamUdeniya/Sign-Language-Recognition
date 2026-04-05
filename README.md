# 🚀 Sign Language Recognition using ResNet50 + Vision Transformer

## 📌 Overview

This project presents a **real-time American Sign Language (ASL) recognition system** built using a hybrid deep learning architecture that combines:

* **ResNet50 (CNN)** for spatial feature extraction
* **Vision Transformer (ViT)** for capturing global contextual relationships

The system enables **live gesture recognition via webcam** and demonstrates high performance on a structured ASL dataset.

---

## 🧠 Model Architecture

```
Input Image
   ↓
ResNet50 (Feature Extraction)
   ↓
Feature Projection
   ↓
Vision Transformer (Global Attention)
   ↓
Classification Layer
```

This hybrid approach leverages:

* CNN → strong local feature learning
* Transformer → global dependency modeling

---

## 🎯 Features

* ✅ Real-time ASL recognition using webcam
* ✅ Hybrid CNN + Transformer architecture
* ✅ High accuracy performance
* ✅ Web-based interface for live detection
* ✅ Modular and scalable codebase

---

## 📊 Performance

* **Validation Accuracy:** 99.93%
* Dataset: ASL Alphabet (preprocessed)

⚠️ Note: Extremely high accuracy may indicate dataset simplicity or limited variability. Real-world performance may vary.

---

## 📂 Dataset

This project uses a **preprocessed version of the ASL Alphabet dataset**, structured into:

* Training set
* Validation set
* Test set

🔗 **Official Dataset (Zenodo DOI):**
https://doi.org/10.5281/zenodo.19427991

👉 Download and extract into project root:

```
ASL_Alphabet_Dataset/
```

📌 Original dataset source:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## 🗂️ Project Structure

```
Sign-Language-Recognition/
│
├── scripts/
│   ├── model.py          # Hybrid CNN + ViT model
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Model evaluation
│   ├── asl_dataset.py    # Data loading & preprocessing
│   └── webcam_test.py    # Real-time prediction
│
├── web/
│   ├── server.py         # Backend server
│   ├── index.html        # UI
│   ├── app.js            # Frontend logic
│   └── styles.css        # Styling
│
├── requirements-web.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
pip install -r requirements-web.txt
```

---

## ▶️ Usage

### 🔹 Run Webcam Detection

```bash
python scripts/webcam_test.py
```

### 🔹 Run Web Application

```bash
python web/server.py
```

Then open in browser:

```
http://127.0.0.1:5000
```

---

## 📌 Key Highlights

* Combines **CNN + Transformer** for improved representation learning
* Demonstrates **end-to-end pipeline** (data → training → deployment)
* Includes both **CLI and web-based inference**

---

## ⚠️ Limitations

* Trained on controlled dataset → may not generalize perfectly to real-world conditions
* Limited variation in lighting/background
* Requires further testing on diverse data

---

## 👥 Authors

* Anupam Udeniya
* Tammay Jain
* Arman Pyrbot

---

## 📚 Citation

If you use this dataset, please cite:

```
ASL Alphabet Dataset (Preprocessed)
https://doi.org/10.5281/zenodo.19427991
```

---

## 🧠 Future Work

* Improve real-world robustness
* Add dynamic gesture recognition (video sequences)
* Deploy as mobile/web application

---
