# Deep Learning Lab Assignments

Complete implementations of all 10 Deep Learning lab assignments.

## Setup

```bash
pip install -r requirements.txt
```

## Labs

| # | Topic | File |
|---|-------|------|
| 1 | PCA – Dimensionality Reduction | `Lab1_PCA/lab1_pca.py` |
| 2 | Simple & Multiple Linear Regression | `Lab2_LinearRegression/lab2_linear_regression.py` |
| 3 | K-Nearest Neighbours (Iris) | `Lab3_KNN/lab3_knn.py` |
| 4 | Naive Bayes Classifier | `Lab4_NaiveBayes/lab4_naive_bayes.py` |
| 5 | Decision Tree + Random Forest + AdaBoost | `Lab5_DecisionTree_RF_AdaBoost/lab5_tree_rf_adaboost.py` |
| 6 | CNN – MNIST Digit Classification | `Lab6_CNN_MNIST/lab6_cnn_mnist.py` |
| 7 | CNN – Plant Disease Detection | `Lab7_CNN_PlantDisease/lab7_plant_disease.py` |
| 8 | CNN – Fashion MNIST Classification | `Lab8_CNN_FashionMNIST/lab8_fashion_mnist.py` |
| 9 | RNN – Stock Price Prediction | `Lab9_RNN_StockPrice/lab9_rnn_stock.py` |
| 10 | LSTM – Weather Prediction | `Lab10_LSTM_Weather/lab10_lstm_weather.py` |

## How to Run

```bash
# Example
cd Lab1_PCA
python lab1_pca.py
```

Labs 1–6, 8–10 use built-in datasets and run without any external data files.  
**Lab 7** requires a plant disease dataset folder (e.g., PlantVillage from Kaggle).

## Notes

- Labs 9 & 10 include synthetic data generators so they run immediately even without a real CSV. Swap in your actual CSV when available (instructions in each file).
- All plots are saved as `.png` in the same folder as the script.
