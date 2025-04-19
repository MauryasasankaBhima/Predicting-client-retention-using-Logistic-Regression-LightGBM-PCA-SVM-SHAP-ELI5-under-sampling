# Predicting-client-retention-using-Logistic-Regression-LightGBM-PCA-SVM-SHAP-ELI5-under-sampling
This project focuses on predicting whether a customer will stay with or leave Santander Bank, using supervised machine learning models, dimensionality reduction techniques, and model explainability tools.

## 🎯 Objective

To accurately identify potential customer churn and support retention strategies by analyzing customer behavior and applying machine learning techniques.

---

## 🛠️ Tools & Techniques

| Task                         | Tools / Techniques                 |
|------------------------------|------------------------------------|
| Classification Models        | Logistic Regression, SVM, LightGBM, Decision Tree |
| Feature Reduction            | PCA (Principal Component Analysis) |
| Data Sampling                | Random Under-sampling              |
| Explainability               | SHAP, ELI5                         |
| Evaluation Metrics           | Accuracy, F1 Score, ROC-AUC        |
| Libraries                    | scikit-learn, lightgbm, shap, eli5, pandas, matplotlib |

---

##  Workflow Summary

1. **Data Preprocessing**
   - Cleaned and standardized customer data
   - Addressed class imbalance using under-sampling

2. **Dimensionality Reduction**
   - Applied PCA to reduce noise and speed up training

3. **Model Training**
   - Trained multiple classifiers: Logistic Regression, SVM, LightGBM, Decision Tree
   - Evaluated models using cross-validation and ROC-AUC

4. **Model Interpretation**
   - Used SHAP and ELI5 to interpret key features affecting client retention

---

##  Sample Results



| Model             | Accuracy | ROC-AUC | F1 Score |
|-------------------|----------|---------|----------|
| LightGBM          | 89.6%    | 0.93    | 0.88     |
| Logistic Regression | 87.2%  | 0.91    | 0.85     |
| SVM               | 85.3%    | 0.89    | 0.82     |

---

## Highlights

- Handled class imbalance effectively with under-sampling
- PCA improved computational efficiency and model performance
- SHAP and ELI5 enhanced interpretability of models
- Compared multiple algorithms for robustness
