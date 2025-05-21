
# 💼 Customer Churn Prediction – End-to-End ML Project

This repository contains an end-to-end machine learning project for predicting customer churn using supervised learning (Random Forest). The project follows a full ML pipeline – from data exploration to deployment-ready model testing.

---

## 📁 Project Structure

```
├── 01_Dependencies.ipynb         # Install & import required libraries
├── 02_EDA.ipynb                  # Exploratory Data Analysis
├── 03_DataPreprocessing.ipynb   # Data cleaning, encoding, feature selection
├── 05_Testing_on_input.ipynb    # Load model & test with sample inputs
├── randomforest_churn_model.pkl # Saved Random Forest model
└── README.md                     # Project documentation
```

---

## 🎯 Objective

The objective of this project is to build a machine learning model that predicts whether a customer will churn (i.e., stop using a service). The solution can be scaled in the future with features like:

* NLP-based customer feedback analysis
* Dashboard for visual analytics
* API and web deployment

---

## 🛠️ Tech Stack

* Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* Jupyter Notebooks
* Random Forest Classifier
* Pickle (model saving)

---

## 🧪 Workflow

### 1. **Dependencies Setup**

* Installed all necessary Python packages.

### 2. **EDA**

* Visualized class balance
* Studied feature distributions & correlations

### 3. **Data Preprocessing**

* Handled missing values
* Applied Label Encoding
* Trained a Random Forest model
* Evaluated model performance (Accuracy: **77.5%**)

### 4. **Model Testing**

* Used the `.pkl` file to test on new, real-time user input

---

## 📈 Model Performance

* **Accuracy:** 77.5%
* **Cross-validation score:** 85%

---

## 🔮 Future Improvements (Phase 2 & 3)

* Add one-hot encoding & feature importance analysis
* Tune hyperparameters with GridSearchCV
* Add NLP-based customer feedback analyzer
* Web deployment (Flask / Streamlit)
* Real-time predictions through API
* Save and use encoders (`encoder.pkl`) for consistency

---

## ✅ Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/Sid-istic/End-to-End-customer-Churn.git
   ```

2. Run the notebooks in order:

   * `01_Dependencies.ipynb`
   * `02_EDA.ipynb`
   * `03_DataPreprocessing.ipynb`
   * `05_Testing_on_input.ipynb`

3. Or run in [Google Colab](https://colab.research.google.com)

---

## 📂 Data

> The dataset used for this project is anonymized and contains customer demographics, service usage, and churn status. (You can mention the source if it's from Kaggle or your own data.)

---

## 🙌 Acknowledgements

This project is part of my machine learning journey. It's designed to evolve over time into a fully deployable product.

---

## 📬 Contact

**Author:** Siddharth Pratap Singh
📧 Email: siddharthsingh10454@gmail.com

---
