from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Function to get a classifier based on the given classifier_name


def get_classifier(classifier_name):
    if classifier_name == 'decision_tree':
        return DecisionTreeClassifier()
    elif classifier_name == 'k_nearest_neighbors':
        return KNeighborsClassifier()
    elif classifier_name == 'perceptron':
        return Perceptron()
    elif classifier_name == 'neural_network':
        return MLPClassifier()
    else:
        raise ValueError('Invalid classifier_name: {}'.format(classifier_name))

# Function to find the minimum set of features that achieve a target accuracy


def find_min_features(classifier_name, X_train, y_train, X_test, y_test, target_accuracy):
    # Get the list of feature names
    feature_names = X_train.columns.tolist()

    # Iterate over all possible combinations of features
    for i in range(1, len(feature_names) + 1):
        for combo in combinations(feature_names, i):
            # Train a model with the current combination of features
            clf = get_classifier(classifier_name)
            clf.fit(X_train[list(combo)], y_train)

            # Evaluate the model on the test set
            accuracy = clf.score(X_test[list(combo)], y_test)

            # If the accuracy is greater than or equal to the target accuracy, return the feature names
            if accuracy >= target_accuracy:
                return list(combo)

    # If no combination of features produces the desired accuracy, return an empty list
    return []

# Function to find the important features using a random forest classifier


def find_important_features(X_train, y_train):
    # Train a random forest classifier on the training data
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Get the feature importance scores and sort them in descending order
    feature_importances = list(zip(X_train.columns, clf.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    # Extract the feature names and return them in a list
    important_features = [f[0] for f in feature_importances]
    return important_features
