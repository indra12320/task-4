# Task 4 - Classification Models and Evaluation

## Objective
Develop and evaluate a binary classification model and understand evaluation metrics and class imbalance.

## Dataset
Breast Cancer Wisconsin dataset available directly from scikit-learn. No separate dataset download is required.

## Models
- Logistic Regression
- Logistic Regression with class weighting
- Decision Tree
- Decision Tree with class weighting

## Evaluation
Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and ROC Curve.

## Handling Imbalanced Data
Class-weighted models use `class_weight='balanced'` to increase the importance of minority-class samples during training.

## Output
A metrics CSV, classification report, summary JSON, four plots, and a saved model file.

## Conclusion
Compare accuracy with precision, recall, F1, and ROC-AUC. When classes are imbalanced, accuracy alone can be misleading; the preferred metric depends on the costs of false positives and false negatives.
