# MainCrafts AI & ML Internship - Task 4

## Classification Models, Evaluation Metrics & Handling Imbalanced Data

This project builds a binary classification system using the built-in Breast Cancer Wisconsin dataset from scikit-learn.

### What it demonstrates
- Binary classification
- Train/test split with stratification
- Logistic Regression and Decision Tree classifiers
- Class imbalance handling using `class_weight="balanced"`
- Confusion matrix
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- ROC curve and comparison plots
- Saving the selected classifier

### Dataset labels
- 0 = Benign
- 1 = Malignant (positive class)

### Run on Windows (PowerShell)
Open PowerShell in this project folder, then run each command:

```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.env\Scripts\Activate.ps1
pip install -r requirements.txt
python task4_classification.py
```

If the environment is already created, activate it and run the script again:

```powershell
.env\Scripts\Activate.ps1
python task4_classification.py
```

### Output files
- `reports/classification_metrics.csv` - metrics comparison of four classifiers
- `reports/classification_report.txt` - detailed report for selected classifier
- `reports/summary.json` - run summary
- `reports/graphs/confusion_matrix_best_model.png`
- `reports/graphs/roc_curve_best_model.png`
- `reports/graphs/model_metrics_comparison.png`
- `reports/graphs/class_distribution.png`
- `artifacts/best_classifier.joblib` - saved selected classifier

### Model selection
The script selects the model with the highest test-set F1 score for this demonstration. In a production workflow, model selection should be performed using validation data or cross-validation before evaluating the final held-out test set.
