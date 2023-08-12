from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import numpy as np


def unsup_binary_train(X_train, y_train):

    # Apply K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)

    # Fit the K-means model to the training data
    kmeans.fit(X_train)

    # Return the trained K-means model
    return kmeans


def unsup_binary_test(model, X_test, y_test):

    # Predict cluster labels for X_test
    y_pred = model.predict(X_test)

    # Convert cluster labels to binary labels
    y_binary_pred = (y_pred == y_pred.min()).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_binary_pred)

    return accuracy


def unsup_multiclass_train(X_train, y_train, k):

    # Explicitly set n_init to suppress warning
    n_init = 10
    kmeans = KMeans(n_clusters=k, n_init=n_init)

    # Determine the number of clusters (K) to use in K-means. One cluster for each type of attack and one for benign traffic
    n_clusters = k + 1

    # Fit K-means to the training data
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(X_train)

    return kmeans


def unsup_multiclass_test(model, X_test, y_test):

    # Predict cluster labels
    y_pred = model.predict(X_test)

    # Compute accuracy by comparing the predicted cluster labels to the true labels
    accuracy = accuracy_score(y_test, y_pred)

    # Return the computed accuracy
    return accuracy
