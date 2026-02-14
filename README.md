# ML-Assignment-2

## Name: Danish Biyabani  
## ID: 2025AA05467  

---

# 1. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset and evaluate their performance using various evaluation metrics. The models are deployed using an interactive Streamlit web application to demonstrate an end-to-end machine learning workflow including model training, evaluation, and deployment.

The chosen task is a **binary classification problem** to predict whether a customer will subscribe to a term deposit based on banking-related attributes.

---

# 2. Dataset Description

- **Dataset Name:** Bank Marketing Dataset  
- **Source:** UCI Machine Learning Repository  
- **Number of Instances:** ~45,000  
- **Number of Features:** 20+  
- **Target Variable:** `y`  
- **Problem Type:** Binary Classification (Yes / No)

The dataset contains demographic and campaign-related attributes such as age, job, marital status, contact type, campaign details, and previous interaction outcomes.

The dataset exhibits **class imbalance**, with approximately 12% positive class samples.

---

# 3. Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

All models were trained using a stratified train-test split to preserve class distribution.

---

# 4. Evaluation Metrics

The following evaluation metrics were computed for each model:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---
# 5. Model Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8457   | 0.9079 | 0.4182    | 0.8147 | 0.5527   | 0.5092 |
| Decision Tree       | 0.8782   | 0.6933 | 0.4785    | 0.4518 | 0.4648   | 0.3963 |
| KNN                 | 0.8986   | 0.8500 | 0.6257    | 0.3318 | 0.4336   | 0.4070 |
| Naive Bayes         | 0.8548   | 0.8101 | 0.4059    | 0.5198 | 0.4559   | 0.3774 |
| Random Forest       | 0.9051   | 0.9286 | 0.6938    | 0.3384 | 0.4549   | 0.4415 |
| XGBoost             | 0.9092   | 0.9341 | 0.6513    | 0.4820 | 0.5540   | 0.5119 |

# 6. Observations on Model Performance

| Model               | Observation                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved high recall due to class balancing, making it effective in identifying positive cases.                           However, precision was relatively lower.
| Decision Tree       | Moderate accuracy but lower AUC score, indicating weaker probability estimation capability.                                              
| KNN                 | High accuracy but low recall, showing sensitivity to class imbalance.                                                                    
| Naive Bayes         | Balanced but moderate performance due to strong feature independence assumptions.                                                        
| Random Forest       | Improved overall accuracy and AUC, but recall remained low for minority class.                                                           |
| XGBoost             | Achieved the best overall performance with highest AUC and MCC, providing a strong balance                                between precision and recall.                 



