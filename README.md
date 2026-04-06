![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Model Training & Evaluation

## Overview

Training a model is only half the job — knowing how to *evaluate* it properly is what separates a notebook experiment from a production-ready pipeline. Poor evaluation practices (testing on training data, ignoring class imbalance, skipping cross-validation) lead to overconfident metrics and models that fail in the real world.

In this lab you will work through the full model evaluation lifecycle: compare splitting strategies, benchmark multiple classifiers with cross-validation, tune hyperparameters with grid search, and interpret the results through confusion matrices, learning curves, and feature importances. You'll use the Digits dataset — 1,797 8×8 grayscale images of handwritten digits (0–9) — which is rich enough to expose interesting evaluation dynamics without requiring GPU resources.

By the end of this lab you will have a repeatable evaluation template that you can apply to any classification problem.

## Learning Goals

By the end of this lab, you should be able to:

- Compare hold-out, k-fold, and stratified k-fold cross-validation and explain when each is appropriate.
- Benchmark multiple classifiers using consistent, multi-metric cross-validation.
- Perform hyperparameter tuning with `GridSearchCV` and interpret the results.
- Diagnose model behavior using confusion matrices, learning curves, and permutation importances.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

The Digits dataset from scikit-learn provides a manageable multiclass classification problem (10 classes, 64 features per sample). Because the classes are roughly balanced, it's a good testbed to understand why stratification still matters and how different models handle a 10-way classification task.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-08-model-training-evaluation.ipynb`**.
2. In the first cell, import everything you'll need:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV,
    learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.inspection import permutation_importance
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations.

## Tasks

### Task 1: Splitting Strategies

1. Load the Digits dataset (`load_digits()`) and convert it to a DataFrame. Examine the class distribution — are the 10 digit classes balanced?
2. Perform a basic **80/20 train/test split** (`random_state=42`). Train a `LogisticRegression(max_iter=5000)` on the training set and report accuracy on the test set.
3. Perform **5-fold cross-validation** (regular `KFold`) with the same model. Report the mean and standard deviation of accuracy.
4. Perform **stratified 5-fold cross-validation** (`StratifiedKFold`). Report the mean and standard deviation of accuracy.
5. Create a bar chart comparing the three accuracy results (single split, k-fold mean, stratified k-fold mean) with error bars for the CV methods.
6. In a markdown cell, answer: Why does stratified splitting matter even when classes are roughly balanced? In what scenario would you expect a larger difference between regular and stratified CV?

### Task 2: Model Comparison with Cross-Validation

1. Define a list of five classifiers:
   - `LogisticRegression(max_iter=5000)`
   - `SVC()`
   - `RandomForestClassifier(random_state=42)`
   - `GradientBoostingClassifier(random_state=42)`
   - `KNeighborsClassifier()`
2. Using **stratified 5-fold CV**, evaluate each model on four metrics: **accuracy**, **precision (macro)**, **recall (macro)**, and **F1 (macro)**.
3. Collect results into a comparison DataFrame with columns: `Model`, `Accuracy (mean ± std)`, `Precision`, `Recall`, `F1`.
4. Sort the DataFrame by F1 score (descending) and display it.
5. In a markdown cell, identify the top 2 models. Are you surprised by the ranking? Which model had the smallest variance across folds?

### Task 3: Hyperparameter Tuning

1. Take the **top 2 models** from Task 2.
2. For each model, define a hyperparameter grid with **at least 3 parameters**. For example:
   - SVC: `{'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto']}`
   - RandomForest: `{'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}`
3. Run `GridSearchCV` with **stratified 5-fold CV** and `scoring='f1_macro'` for each model.
4. Report the **best parameters** and **best cross-validation score** for each model.
5. Compare the **default** (Task 2) vs. **tuned** performance. Did tuning make a meaningful difference?
6. In a markdown cell, discuss: Was the improvement worth the computational cost? How would you approach tuning if the dataset were 100× larger?

### Task 4: Interpretation & Diagnostics

Using the **best tuned model** from Task 3:

1. Train it on the full training set and predict on the test set.
2. Plot the **confusion matrix** (10×10 for digits 0–9) using `ConfusionMatrixDisplay`. Use a color map that makes misclassifications easy to spot.
3. Plot **learning curves**: vary the training set size from 10% to 100% and plot both training and validation scores. Use `sklearn.model_selection.learning_curve`.
4. Compute **permutation importances** on the test set. Plot the top 20 features by importance (with error bars).
5. In a markdown cell, write a summary addressing:
   - Is the model overfitting, underfitting, or well-fitted? (Use the learning curves as evidence.)
   - Which digit pairs are most commonly confused? Why might that be?
   - Which pixel positions (features) matter most? Does that make intuitive sense?
   - What would you try next to improve performance?

## Submission

### What to submit
- `m4-08-model-training-evaluation.ipynb` — completed notebook with all four tasks.

### Definition of done (checklist)
- [ ] Three splitting strategies are compared with a bar chart.
- [ ] Five models are benchmarked in a comparison DataFrame sorted by F1.
- [ ] Top 2 models are tuned with GridSearchCV; default vs. tuned performance is compared.
- [ ] Confusion matrix, learning curves, and permutation importances are plotted and interpreted.
- [ ] Every task includes at least one markdown cell with interpretation.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete model training and evaluation"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
