# 💳 Credit Card Fraud Detection using Machine Learning

This project applies various machine learning models to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud datasets, the project emphasizes preprocessing, resampling, and model evaluation to find the most effective solution for this real-world classification problem.

## 🧠 Project Overview

The goal is to identify fraudulent credit card transactions based on a variety of anonymized features. Since fraud cases are extremely rare compared to normal transactions, addressing **class imbalance** was a key part of the process.

**Key Steps:**

1. **Data Preprocessing**  
   - Removed irrelevant features  
   - Scaled data for model compatibility  
   - Analyzed feature correlations and class distributions

2. **Baseline Model Testing**  
   - Trained multiple ML models (e.g., Logistic Regression, Random Forest, Decision Tree, etc.)  
   - Evaluated performance metrics (accuracy, precision, recall, F1-score)

3. **Class Imbalance Handling**  
   - Discovered that fraud cases made up a small fraction of the dataset  
   - Applied **oversampling** techniques to balance the dataset (e.g., SMOTE or Random Oversampling)

4. **Final Model Testing**  
   - Re-evaluated all models on the balanced data  
   - Found that a **Sequential Neural Network** (Keras) outperformed other methods

## 📊 Dataset

The dataset is a public credit card transactions dataset. It contains:

- 284,807 transactions
- 492 fraud cases (~0.172%)
- 30 numerical features (anonymized PCA components + Time and Amount)

## 🛠️ Technologies Used

- Python 3  
- Pandas & NumPy  
- Scikit-learn  
- TensorFlow & Keras  
- Matplotlib

## 📈 Model Evaluation Metrics

Given the class imbalance, accuracy was not a reliable metric. Instead, the following metrics were prioritized:

- **Precision** – How many predicted frauds were actual frauds
- **Recall** – How many actual frauds were correctly predicted
- **F1-Score** – Harmonic mean of precision and recall
- **AUC-ROC** – Performance across different thresholds

## 🧪 Final Model

In the end, it was conclusive that the best model for this application was the **Sequential Neural Network** with:

- Input layer matching feature shape  
- Hidden dense layers with ReLU activations  
- Dropout layers to reduce overfitting  
- Output sigmoid layer for binary classification

This model achieved **significantly higher precision and recall** on the balanced dataset compared to traditional ML models.

## 🚀 How to Run

The best way to run this file is to open it directly in Google Colab to utilize the powerful GPUs and TPUs through a virtual runtime for free to reduce processing times as these models are very demanding.
