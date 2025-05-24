# Customer Churn Prediction – Model Optimization Pipeline 🚀

This repository contains a complete pipeline for feature engineering, model refinement, and hyperparameter tuning on a customer churn dataset. The main goal was to improve the model's performance and interpretability step-by-step.

---

## 📁 Files Overview

### 1. `01_Feature_Engineering.ipynb`

- **Purpose**: Performed advanced feature engineering.
- **What was done**:
  - Created a new feature: `Tenure_Monthly` = `tenure` × `MonthlyCharges`.
  - Applied **LabelEncoding** to categorical columns.
  - Trained a baseline RandomForest model using the new encoded dataset.
- **Outcome**: Slight improvement in model performance and better feature representation.

---

### 2. `02_FeatureImportance.ipynb`

- **Purpose**: Analyze feature importances and reduce dimensionality.
- **What was done**:
  - Used the one-hot encoded dataset from step 1.
  - Identified the most impactful features using `.feature_importances_` from the trained model.
  - Dropped all features with importance < `0.03`.
  - Saved the reduced and optimized dataset as `new_training_data.csv`(can be found in data folder).
- **Outcome**: Cleaner dataset with less noise and better interpretability.

---

### 3. `03_model_optimization.ipynb`

- **Purpose**: Tune the model using hyperparameter optimization.
- **What was done**:
  - Used `RandomizedSearchCV` on the reduced feature set.
  - Tuned key parameters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `bootstrap`.
  - Trained and evaluated the optimized model which was saved by the name `best_model.pkl` and is available in models folder.
- **Final Accuracy**:
  - **Training Accuracy**: with best parameters➝ **84.86%**
  - **Test Accuracy**: Increased from **77.39%** ➝ **77.76%** after feature Engineering
  - **Test Accuracy**: Increased from **77.76** ➝ **80.41%** after model optimization

---

## ✅ Summary

- 🧠 Feature Engineering boosted model capacity.
- 🗂️ Feature Selection reduced noise and improved generalization.
- ⚙️ Hyperparameter Tuning significantly improved performance.

This pipeline demonstrates how small, incremental improvements in the ML workflow can lead to a substantial increase in prediction accuracy.

---

## 📌 Requirements

- Python ≥ 3.8  
- scikit-learn  
- pandas  
- numpy  
- matplotlib / seaborn (for visualizations)  
- shap (optional, for feature explanation)

---

## 📬 Contact

For questions or suggestions, feel free to reach out.

---

