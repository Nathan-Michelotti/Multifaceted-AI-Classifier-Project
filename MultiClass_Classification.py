from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Function to train a direct multiclass classification model


def direct_multiclass_train(model_name, X_train, y_train):
    if model_name == "dt":
        model = DecisionTreeClassifier()
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "perceptron":
        model = Perceptron()
    elif model_name == "nn":
        model = MLPClassifier()
    else:
        raise ValueError("Invalid model name")

    # Training the model
    model.fit(X_train, y_train)
    return model

# Function to test a direct multiclass classification model


def direct_multiclass_test(model, X_test, y_test):
    # Making predictions on the test set
    predictions = model.predict(X_test)
    # Calculating accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Function to train a binary classification model for benign vs malicious


def benign_mal_train(model_name, X_train, y_train):
    # Converting the target variable to binary classes ("benign" or "malicious")
    y_train_binary = y_train.apply(
        lambda x: "benign" if x == "benign" else "malicious")
    # Training the binary classification model
    return direct_multiclass_train(model_name, X_train, y_train_binary)

# Function to test a binary classification model for benign vs malicious


def benign_mal_test(model, X_test):
    # Making predictions on the test set
    return model.predict(X_test)

# Function to train a binary classification model for malicious samples only


def mal_train(model_name, X_train, y_train):
    # Filtering the training data to include only malicious samples
    X_train_mal = X_train[y_train != "benign"]
    y_train_mal = y_train[y_train != "benign"]
    # Training the binary classification model for malicious samples
    return direct_multiclass_train(model_name, X_train_mal, y_train_mal)

# Function to test a binary classification model for malicious samples only


def mal_test(model, X_test):
    # Making predictions on the test set
    return model.predict(X_test)

# Function to evaluate hierarchical predictions


def evaluate_hierarchical(benign_preds, mal_preds, y_test):
    final_preds = []
    for b, m in zip(benign_preds, mal_preds):
        # Combining predictions from benign and malicious models
        final_preds.append(m if b == "malicious" else "benign")

    # Calculating accuracy of the hierarchical predictions
    accuracy = accuracy_score(y_test, final_preds)
    return accuracy
