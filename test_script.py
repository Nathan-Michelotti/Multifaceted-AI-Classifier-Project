import pandas as pd
import MultiClass_Classification as mcc
import Feature_Selection as fs
import Unsupervised_Learning as ul
import helpers as help
import Regression as reg

# Load and clean the data
df = help.load_data("File1.csv")
df = help.format_column_names(df)
df_clean = help.clean_data(df, "replace")
X_train, y_train, X_test, y_test = help.split_data(df_clean, "Label")

# Multi-Class Classification
# Train and test a direct multi-class classification model using Decision Tree
model = mcc.direct_multiclass_train("dt", X_train, y_train)
acc = mcc.direct_multiclass_test(model, X_test, y_test)
print("MCC Direct Accuracy:", acc)

# Train a binary classification model for benign vs malicious using Decision Tree
model = mcc.benign_mal_train("dt", X_train, y_train)
benign_preds = mcc.benign_mal_test(model, X_test)

# Train a binary classification model for malicious samples only using Decision Tree
model = mcc.mal_train("dt", X_train, y_train)
mal_preds = mcc.mal_test(model, X_test)

# Evaluate hierarchical predictions combining benign and malicious models
acc = mcc.evaluate_hierarchical(benign_preds, mal_preds, y_test)
print("MCC Hierarchical Accuracy:", acc)

# Feature Selection
# Find the minimum set of features that achieve 90% accuracy using Decision Tree
features = fs.find_min_features(
    "decision_tree", X_train, y_train, X_test, y_test, 0.9)
print("Min Features:", features)

# Find the important features using a decision tree classifier
features = fs.find_important_features(X_train, y_train)
print("Feature Ordering:", features)

# Unsupervised Learning
# Train and test an unsupervised binary classification model
model = ul.unsup_binary_train(X_train, y_train)
acc = ul.unsup_binary_test(model, X_test, y_test)
print("Unsupervised Binary Acc:", acc)

# Train and test an unsupervised multi-class classification model with 3 clusters
model = ul.unsup_multiclass_train(X_train, y_train, 3)
acc = ul.unsup_multiclass_test(model, X_test, y_test)
print("Unsupervised Multi-Class Acc:", acc)

# Regression
# Train a benign regression model using Linear Regression
model = reg.benign_regression_train("linear_regression", X_train, y_train)
thresh = reg.benign_regression_test(model, X_test, y_test)
acc = reg.benign_regression_evaluate(model, X_test, y_test, thresh)
print("Regression Threshold:", thresh)
print("Regression Accuracy:", acc)
