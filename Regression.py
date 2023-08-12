from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# Function to train a benign regression model


def benign_regression_train(model_name, X_train, y_train):
    if model_name == 'linear_regression':
        # Create and train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model name")

    return model

# Function to test a benign regression model


def benign_regression_test(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate threshold as the square root of MSE
    threshold = np.sqrt(mse)
    return threshold

# Function to convert multi-class labels to binary labels


def convert_to_binary_labels(y, benign_label):
    # Convert y to binary labels where benign_label is 0 and other labels are 1
    return np.where(y == benign_label, 0, 1)

# Function to evaluate a benign regression model


def benign_regression_evaluate(model, X_test, y_test, threshold):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate mean squared error (MSE) and root mean squared error (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Convert RMSE values to binary predictions using the threshold
    y_pred_binary = np.where(rmse > threshold, 1, 0)

    # Assuming y_test has multi-class labels, convert it to binary labels (benign=0, malicious=1)
    benign_label = "Benign"
    y_test_binary = convert_to_binary_labels(y_test, benign_label)

    # Calculate accuracy score based on binary predictions
    accuracy = accuracy_score(y_test_binary, y_pred_binary)

    return accuracy
