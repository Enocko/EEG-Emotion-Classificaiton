
# EEG-Based Emotion Classification Using Spectral Features and Machine Learning

## 🧠 Project Overview

This project investigates the use of EEG (electroencephalogram) spectral features to predict emotional states using machine learning models. The goal is to identify which frequency-domain features are most discriminative and build a robust, modular pipeline for training, evaluating, and predicting emotional states (e.g., Positive, Neutral, Negative).

> **Research Question:**  
> *Can machine learning models predict emotional states from EEG spectral features, and which features are most discriminative for emotional prediction?*

---

## 🔍 Key Features

- 📊 Loads EEG dataset with thousands of spectral features  
- 🧼 Preprocesses data (e.g., handling nulls, encoding labels)
- 🤖 Trains multiple classifiers:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - K-Nearest Neighbors
  - Gradient Boosting
  - Simple Deep Learning (optional)
- 🏆 Compares model performance and selects the best
- 📈 Visualizes:
  - Accuracy comparisons
  - Confusion matrix
  - Feature importance
- 💾 Saves best model for future predictions
- 🔮 Predicts new emotional states on unseen EEG data

---

## 📁 Project Structure

```

eeg-emotion-classification/
├── data/
│   └── emotions.csv             # Original EEG dataset
├── models/
│   └── model.pkl                # Saved best ML model
├── notebooks/                   # Jupyter exploration notebooks
│   └── analysis.ipynb
├── src/
│   ├── load\_data.py             # Dataset loader
│   ├── preprocess.py            # Preprocessing logic
│   ├── train\_models.py          # Training and evaluation
│   ├── visualize.py             # Plotting and charts
│   └── predict.py               # Inference on new data
├── main.py                      # End-to-end pipeline runner
├── requirements.txt             # Dependencies
└── README.md                    # You’re here

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/eeg-emotion-classification.git
cd eeg-emotion-classification
````

### 2. Create Virtual Environment

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

Place your EEG dataset here:

```bash
data/emotions.csv
```

> Ensure:
>
> * Label column is named `label`
> * All other columns are numerical EEG features

---

## 🚀 Run the Pipeline

### Full Training & Evaluation

```bash
python main.py
```

This will:

* Load and preprocess data
* Train and evaluate all models
* Visualize performance and feature importances
* Save best model to `models/model.pkl`

---

## 🔮 Predict on New Data

Use the trained model to predict emotional states:

```bash
python src/predict.py --file data/new_eeg_data.csv
```

> Your new file should follow the same format (no label column, only features)

---

## 📊 Visualizations

* **Accuracy Bar Chart** — easily compare all model accuracies
* **Confusion Matrix** — highlights classification strengths and weaknesses
* **Feature Importance (Random Forest)** — understand what brain signals matter most

---

## 📌 Sample Results

| Model                   | Accuracy | Precision | Recall | F1-score |
| ----------------------- | -------- | --------- | ------ | -------- |
| Gradient Boosting       | 99.8%    | 99.7%     | 99.8%  | 99.8%    |
| Random Forest           | 98.8%    | 98.8%     | 98.8%  | 98.8%    |
| Deep Learning (reduced) | 90.6%    | N/A       | N/A    | N/A      |
| Logistic Regression     | 42.2%    | 31.3%     | 42.2%  | 34.2%    |
| SVM                     | 38.4%    | 35.9%     | 38.4%  | 25.0%    |

---

## 📚 Research Insights

* The top EEG features (e.g., `min_q_5_b`, `mean_d_15_b`, `fft_742_b`) were dominant in emotion prediction.
* Random Forest and Gradient Boosting consistently outperformed other models.
* Dimensionality reduction helped improve Logistic Regression and Deep Learning models.
* Deep learning models show potential but need further optimization.

---

## 📌 Notes

* The `data/`, `models/`, and `venv/` folders should be excluded from GitHub using `.gitignore`
* Extend model training in `train_models.py` to add custom ML/DL models
* For deep learning, consider using PyTorch or TensorFlow with hyperparameter tuning



