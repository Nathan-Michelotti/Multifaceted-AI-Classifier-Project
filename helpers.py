import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load data from a CSV file


def load_data(fname):
    df = pd.read_csv(fname)
    return df

# Function to format column names by replacing spaces, slashes, and dashes with underscores


def format_column_names(df):
    df.columns = df.columns.str.strip().str.replace(
        ' ', '_').str.replace('/', '').str.replace('-', '_')
    return df

# Function to clean data based on a flag ("replace" or "remove")


def clean_data(df, flag):
    if flag == "replace":
        # Replacing NaN values with the mean of the column
        df.fillna(df.mean(), inplace=True)
    elif flag == "remove":
        # Dropping rows with NaN values
        df.dropna(inplace=True)

    # Removing non-numerical columns
    df = df.select_dtypes(include=[np.number])

    return df

# Function to split data into training and testing sets


def split_data(df, label):
    X = df.drop(label, axis=1)
    y = df[label]

    # Splitting the data into 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test
