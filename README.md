# Toxic-Comment-Classification-MultiLabel
A deep learning model for multi-label classification of toxic comments using focal loss

This project is built as part of the IIT Madras Data Science coursework.

## Problem Statement
Develop a deep learning model to classify comments into multiple toxicity labels:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The model performs **multi-label classification** using Keras with a custom **Focal Loss** function to address class imbalance.

## Dataset
The dataset used is from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).  
It contains comments and six target toxicity categories.

## Methodology
- Text Preprocessing using Keras Tokenizer and Padding
- Model: Deep Neural Network with Embedding + Dense Layers
- Loss Function: Custom Focal Loss (handles imbalance)
- Evaluation Metrics: Accuracy, Validation Loss, and `classification_report` using `sklearn`
- Visualizations: Accuracy/Loss per Epoch

## Results
The model achieves near-perfect recall but very low precision due to dataset imbalance — improved using Focal Loss.

## Visualizations
Training and validation accuracy and loss graphs plotted using `matplotlib`.

## Requirements
- TensorFlow / Keras
- NumPy
- scikit-learn
- matplotlib

Install with:
```bash
pip install tensorflow scikit-learn matplotlib
