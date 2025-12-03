# SDUWPS-ML-Classifier
Stress Detection Using Wearable Physiological Signals (SDUWP): A robust project for detecting stress and physical effort using physiological signals collected from wearable devices. This project implements a complete pipeline including raw data preprocessing, feature extraction with tsfresh, and ensemble classification using Random Forest, XGBoost, and LightGBM.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20XGBoost%20%7C%20tsfresh-orange)
![License](https://img.shields.io/badge/License-MIT-green)
---

## 📋 Table of Contents
- [Authors](#-authors)
- [Paper & Documentation](#-Paper-&-Documentation)
- [About the Project](#-about-the-project)
- [Dataset & Sensors](#-dataset--sensors)
- [Methodology Pipeline](#-methodology-pipeline)
  - [Preprocessing](#1-preprocessing-and-cleaning)
  - [Feature Engineering](#2-feature-engineering-tsfresh)
  - [Machine Learning Models](#3-machine-learning-models)
- [Results](#-results-and-performance)
- [Project Structure](#-project-structure)

---

## 👥 Authors

* **Anderson Cristiano Sassaki Gonçalves** - andersoncsg@estudante.ufscar.br

* **Lorenzo Grippo Chiachio** - lorenzo.chiachio@estudante.ufscar.br

---

## 📄 Paper & Documentation

A comprehensive scientific report detailing the theoretical background, methodology, and in-depth experimental analysis is available in the `RELATORIO` directory.

* **Title:** SDUWPS: Detecção de Estresse e Esforço Físico com Sinais Fisiológicos
* **Language:** Portuguese (PT-BR)
* **Format:** PDF & LaTeX Source
* **Location:** [`RELATORIO/`](./RELATORIO)

The paper provides a deeper dive into the specific challenges of distinguishing physiological stress from physical exertion and justifies the architectural choices made in this pipeline (e.g., why demographics were excluded and the impact of the `tsfresh` feature selection).

> **Paper Title:** SDUWPS: Detecção de Estresse e Esforço Físico com Sinais Fisiológicos
> **Authors:** Anderson Cristiano Sassaki Gonçalves & Lorenzo Grippo Chiachio
> **Institution:** Federal University of São Carlos (UFSCar) - Sorocaba Campus

---

## 📖 About the Project

**SDUWPS** is a Machine Learning framework designed to classify physiological states of the human body using raw data from wearable devices. The primary goal is to distinguish between three distinct states based on multivariate time-series data:

1.  **RELAXATION/STRESS** (Cognitive/Psychological state)
2.  **AEROBIC ACTIVITY** (Physical exertion)
3.  **ANAEROBIC ACTIVITY** (Intense physical exertion)

The project explores the challenge of separating physiological arousal caused by mental stress from arousal caused by physical exercise, utilizing a pipeline that includes advanced signal processing, automated feature extraction, and ensemble learning algorithms.

---

## 📡 Dataset & Sensors

The model is trained on a dataset containing raw time-series data collected from wearable sensors during structured experimental sessions. The input features include:

* **ACC:** Triaxial Accelerometer (Motion intensity).
* **EDA:** Electrodermal Activity (Skin conductance/Sweat).
* **TEMP:** Body Temperature (Skin temperature).
* **HR:** Heart Rate.
* **IBI:** Inter-beat Intervals (Heart Rate Variability).
* **BVP:** Blood Volume Pulse.

> **Note:** Demographic data (Age, Gender, Weight) was analyzed and also included on the training stage, even though it didn't have a considerate impact on the final model's precision.

---

## ⚙️ Methodology Pipeline

The solution is implemented in Python and follows a rigorous Data Science pipeline:

### 1. Preprocessing and Cleaning
* **Outlier Removal:** Applied Hard Clipping based on statistical thresholds ($2 \times IQR$).
* **Imputation:** Conditional mean imputation for specific missing sensor data.
* **Normalization:** Scaling of sensor readings to ensure magnitude consistency.

### 2. Feature Engineering (`tsfresh`)
We utilized **[tsfresh](https://tsfresh.readthedocs.io/)** (Time Series Feature extraction on basis of Scalable Hypothesis tests) to automatically extract hundreds of mathematical characteristics from the raw signals.
* **Extraction:** Statistical, spectral, and temporal features.
* **Selection:** Hypothesis testing was used to filter out irrelevant features, keeping only those with high discriminative power for the target classes.

### 3. Machine Learning Models
A benchmark was performed comparing classic algorithms against state-of-the-art Ensemble methods:
* **Baselines:** KNN, Naive Bayes, Logistic Regression, SVM, MLP.
* **Ensembles:** Random Forest (Bagging).
* **Gradient Boosting:** XGBoost and LightGBM.

Dimensionality reduction techniques (**PCA** and **t-SNE**) were also employed for data visualization and cluster analysis.

---

## 📊 Results and Performance

The experiments demonstrated that the **XGBoost** model achieved the best overall performance. Key findings include:

* **High Accuracy:** The model successfully distinguishes between Stress and Physical Activity with high precision.
* **Stress Detection:** Achieved an F1-Score close to **1.0** for the `STRESS` class, proving the system's reliability for health monitoring.
* **Feature Importance:** Data from the **tags** (temporal alignment) combined with **ACC** and **EDA** features proved to be the most critical predictors.
* **Demographics:** Including demographic data resulted in lower generalization performance, confirming the decision to rely solely on sensor dynamics.

---

## 📂 Project Structure

```bash
SDUWPS-Stress-Detection/
├── IMPLEMENTACAO
    ├── dataset/                # Raw sensor data and CSV files
    │   ├── train.csv           # Training labels
    │   ├── users_info.txt      # Demographic metadata
    │   └── ...
    ├── figs/                   # Generated plots (Confusion Matrix, ROC, etc.)
    ├── scripts/                # Helper Python modules
    │   ├── experimentos.py     # Model training and benchmarking functions
    │   └── preprocessamento.py # Cleaning and Feature Extraction functions
    ├── main.ipynb              # Main Jupyter Notebook (Run this file)
├── RELATORIO                   # Directory that contains the paper for this project and its .tex code and images
└── README.md                   # Project documentation
```
