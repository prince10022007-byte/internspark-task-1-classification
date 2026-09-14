# InternSpark Task 1 — Supervised Classification

## Project: Breast Cancer Wisconsin Classification

### Objective
Build and evaluate supervised machine learning classification models with preprocessing, train/test split, 5-fold cross-validation, and multiple evaluation metrics.

### Dataset
The project uses the Breast Cancer Wisconsin dataset from scikit-learn.

- Samples: 569
- Features: 30 numeric features
- Classes: Malignant and Benign

### Preprocessing
- Stratified 80/20 train-test split
- StandardScaler for Logistic Regression
- 5-fold Stratified Cross-Validation
- Random state: 42

### Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation Accuracy

### Results

| Model | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 |

### Best Model
Logistic Regression achieved the best overall test performance and ROC-AUC score.

### Files
- `Task_1_Supervised_Classification.ipynb` — Complete machine learning notebook
- `Task_1_Short_Report_Complete.docx` — Project report
- `requirements.txt` — Required Python packages
- `results.csv` — Model evaluation results

### Conclusion
The project demonstrates a complete supervised classification workflow including data preprocessing, model training, cross-validation, evaluation, and comparison of multiple machine learning algorithms.
