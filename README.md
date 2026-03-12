# Sign Language Recognition using ResNet50 + Vision Transformer

## Overview

This project implements a **real-time American Sign Language (ASL) recognition system** using a hybrid deep learning architecture combining **ResNet50 (CNN)** and **Vision Transformer (ViT)**.

The model extracts spatial features using ResNet50 and applies transformer-based attention to capture global relationships between features.

## Features

* Real-time ASL recognition
* Hybrid CNN + Transformer architecture
* Webcam-based prediction
* Web interface for live detection
* High accuracy model

## Model Architecture

ResNet50 → Feature Projection → Vision Transformer → Classification Layer

## Accuracy

**99.93% validation accuracy** on the ASL Alphabet dataset.

## Dataset

ASL Alphabet Dataset

Download:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Project Structure

```
Sign-Language-Recognition/
│
├── scripts/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── asl_dataset.py
│   └── webcam_test.py
│
├── web/
│   ├── server.py
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── requirements-web.txt
├── .gitignore
└── README.md
```

## Installation

```
pip install -r requirements-web.txt
```

## Run Webcam Detection

```
python scripts/webcam_test.py
```

## Run Web Application

```
python web/server.py
```
## Web Demo
```
Run the web application locally:

python web/server.py

Then open in browser:
http://127.0.0.1:5000
```
## Author

Anupam Udeniya
