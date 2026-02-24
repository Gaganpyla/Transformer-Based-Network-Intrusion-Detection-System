# 🛡️ Transformer-Based Network Intrusion Detection System (NIDS)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Linformer](https://img.shields.io/badge/Architecture-Linformer-green.svg)](https://github.com/lucidrains/linformer)

## 📌 Project Overview
This project implements a high-performance **Network Intrusion Detection System (NIDS)** using a **Transformer-based architecture**. Specifically, it utilizes the **Linformer**—a linear-complexity variant of the standard Transformer—to efficiently process large-scale network traffic data. 

The system performs binary classification to distinguish between **Normal** traffic and **Attack** traffic, leveraging modern deep learning techniques to enhance cybersecurity defenses.

## 📊 Dataset Specifications
The model is trained and evaluated across two major cybersecurity benchmarks:
1.  **UNSW-NB15 (Primary):** Used for training and initial testing. It contains a diverse range of modern network attack patterns.
    * **Training Samples:** 175,341
    * **Testing Samples:** 82,332
2.  **NSL-KDD (Generalization):** Used as an unseen dataset to evaluate the model's ability to generalize to different network environments.

## 🛠️ Implementation Workflow

### 1. Preprocessing
* **Label Encoding:** Binary labels are encoded as `0` for Normal and `1` for Attack.
* **Data Cleaning:** Automated handling of infinite (`inf`) and null (`NaN`) values to ensure data integrity.
* **Categorical Encoding:** Non-numeric features (e.g., protocol, service, state) are converted using `LabelEncoder`.

### 2. Normalization
* **Feature Scaling:** All numerical features are standardized using `StandardScaler` based on training set statistics to ensure uniform input distribution.

### 3. Model Architecture
* **Linformer Layer:** The core transformer block reduces self-attention complexity from $O(n^2)$ to $O(n)$, allowing for efficient scaling to high-dimensional network data.
* **Classification Head:** A robust Multi-Layer Perceptron (MLP) serves as the final decision layer for binary classification.

### 4. Training Enhancements
* **Mixed Precision:** Utilizes `torch.amp` (autocast) and `GradScaler` to significantly speed up training on GPUs while reducing memory footprint.
* **Class Weighting:** Implements `balanced` class weights to address data imbalance, ensuring the model remains sensitive to minority attack classes.

## 📈 Performance Metrics

### UNSW-NB15 Results
The model achieves high detection accuracy and a balanced F1-score on the native test set:
* **Accuracy:** ~88.52%
* **F1-Score (Attack):** 0.90
* **Precision (Normal):** 0.90

### Generalization (NSL-KDD) Results
When tested on the unseen NSL-KDD dataset without further fine-tuning:
* **Accuracy:** ~73.71%
* **F1-Score:** ~0.74

## 🚀 Usage
Open the `Transformer-Based Network Intrusion Detection System.ipynb` in **Jupyter Notebook** or **Google Colab**. Ensure the dataset `.parquet` files (UNSW_NB15 training/testing sets) are in the working directory and run the cells sequentially to perform:
1.  **Data Preprocessing & EDA:** Clean data and visualize class distributions.
2.  **Model Training:** Train the Linformer model (CUDA supported for performance).
3.  **Performance Evaluation:** Generate classification reports and Confusion Matrix visualizations.

## 📝 License
This project is provided under the terms found in the accompanying [LICENSE](LICENSE) file.

---
**Disclaimer:** *This system is intended for research and educational purposes in the field of cybersecurity.*
