# Automated Material Stream Identification (MSI) System ♻️

## 📌 Project Overview
The efficient sorting of post-consumer waste is a critical bottleneck in achieving circular economy goals. This project is an **Automated Material Stream Identification (MSI) System** designed to classify waste items into specific categories using fundamental Machine Learning techniques.

This project focuses on **feature-based computer vision**, mastering the pipeline of Data Augmentation, Feature Extraction (HOG + Color Histograms), and Classification (SVM vs. k-NN).

## 🗂️ Material Classes
The system classifies objects into **7 categories**:
1. **Glass** 🍾
2. **Paper** 📄
3. **Cardboard** 📦
4. **Plastic** 🥤
5. **Metal** 🥫
6. **Trash** 🗑️
7. **Unknown** ❓ (Rejection mechanism for low-confidence predictions)

## ⚙️ Methodology

### 1. Data Preprocessing & Augmentation
To ensure robust training and meet the project requirement of a **30% dataset increase**:
* **Resizing:** All images are normalized to fixed dimensions (`128x64`).
* **Augmentation:** * **Horizontal Flipping:** Doubles the dataset size.
    * **Rotation (±15°):** Adds robustness against orientation changes.

### 2. Feature Extraction
We convert raw images into 1D numerical feature vectors using a hybrid approach:
* **Histogram of Oriented Gradients (HOG):** Captures the **shape** and edge structure (e.g., the outline of a bottle vs. a can).
* **Color Histograms:** Captures **color distribution** (e.g., distinguishing brown cardboard from white paper).
* **Vector Stacking:** These features are concatenated (`np.hstack`) to form a complete "fingerprint" of the object.

### 3. Classification Models
Two foundational classifiers were implemented and compared:
* **Variant A: Support Vector Machine (SVM):** Uses an RBF kernel for non-linear separation.
* **Variant B: k-Nearest Neighbors (k-NN):** Uses distance-based weighting.

**Unknown Handling:** A probability threshold (e.g., 0.6) is applied. If the model's confidence is below this threshold, the item is classified as "Unknown" to prevent false positives.
