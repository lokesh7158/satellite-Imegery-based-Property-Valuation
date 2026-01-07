
# 🏡 satellite-Imegery-based-Property-Valuation

**Satellite Imagery + Tabular Data Regression**

## 📌 Overview

This project implements a **multimodal regression pipeline** to predict residential property prices by combining **structured housing attributes** with **satellite imagery**.
The core idea is to enhance traditional real estate valuation models by incorporating **environmental and neighborhood context**—such as green cover, road density, and surrounding infrastructure—captured directly from satellite images.

Unlike standard price prediction systems that rely only on numerical features, this approach fuses **tabular data** with **visual embeddings** extracted from satellite images using deep learning.



## 🎯 Objectives

* Build a **multimodal regression model** to predict house prices.
* Programmatically fetch **satellite images** using latitude and longitude coordinates.
* Extract **high-level visual features** from images using a CNN.
* Combine **image features + tabular features** into a single predictive model.
* Compare performance between:

  -> Tabular-only model
  -> Multimodal (Tabular + Image) model
* Provide **model explainability** using Grad-CAM to visualize influential image regions.



## 📊 Dataset

### 1. Tabular Data


* **Key Features:**

  * `price` (target)
  * `bedrooms`, `bathrooms`
  * `sqft_living`, `sqft_lot`
  * `grade`, `condition`, `view`
  * `waterfront`
  * `lat`, `long` (used for satellite image retrieval)

### 2. Satellite Imagery

* Satellite images are fetched using **latitude & longitude** coordinates.
* Images capture surrounding environmental context such as:

  * Vegetation
  * Water bodies
  * Urban density
* **Source:** Sentinel-2 imagery via Sentinel Hub API.



## Satellite Image Acquisition

* Images are programmatically downloaded using **Sentinel Hub Process API**.
* Each property is mapped to a **bounding box (~500m radius)** around its coordinates.
* Cloud-free mosaics are selected automatically.
* RGB images are generated using Sentinel-2 spectral bands.

The image fetching logic is implemented in:


data_fetcher.py


---

## 🧠 Model Architecture

### Tabular Branch

* Fully connected neural network
* Handles numerical housing attributes

### Image Branch

* Convolutional Neural Network (CNN)
* Extracts dense visual embeddings from satellite images

### Fusion Strategy

* Image embeddings are concatenated with tabular features
* Joint regression head predicts final property price

---



## 📁 Project Structure

```
├── data_fetcher.py           # Satellite image download pipeline
├── preprocessing.ipynb       # Data cleaning & feature engineering
├── model_training.ipynb      # Multimodal model training
├── train_raw.xlsx            # Raw training data
├── test.xlsx                 # Raw test data
├── metadata_train.csv        # Train data with image paths
├── metadata_test.csv         # Test data with image paths
├── processed_train.csv       # Final training features
├── processed_test.csv        # Final test features
├── images/                   # Downloaded satellite images
└── README.md
```

---

## ⚙️ Setup & Installation

### Requirements

* Python 3.8+
* Libraries:

  ```
  numpy
  pandas
  matplotlib
  scikit-learn
  torch / tensorflow
  requests
  opencv-python
  ```

Install dependencies:


pip install -r requirements.txt


---

## ▶️ How to Run

### 1. Fetch Satellite Images

->Run 
 data_fetcher.py


### 2. Preprocess Data

Open and run:


preprocessing.ipynb


### 3. Train the Model

Open and run:


model_training.ipynb




## 📈 Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **R² Score**

Model performance is compared between:

* Tabular-only regression
* Multimodal regression (Tabular + Image)

---

## 📊 Results Summary

| Model            | RMSE ↓       | R² ↑       |
| ---------------- | ------------ | ---------- |
| Tabular Only     | 0.2124       | 0.8388     |
| Tabular + Images | 0.27         | 0.73       |

tabular only performs better.
---

## 📦 Deliverables

* 📄 `predictions.csv` — final predicted prices
* 🧠 Trained multimodal regression model
* 📊 Visual & numerical performance comparison
* 🛰 Satellite image dataset linked to properties

---

## 🚀 Key Takeaways

* Satellite imagery provides **valuable contextual signals** for property valuation.
* Visual explainability builds trust in deep learning-based valuation systems.


