import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Funktion 1: Data Cleaning
def clean_data(df):
    """
    Clean the input dataset by filling missing values only for numeric columns.
    Non-numeric columns are left as they are.
    """
    # Fill missing values for numeric columns with the mean
    df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number])  # Non-numeric columns
    
    df_numeric_cleaned = df_numeric.fillna(df_numeric.mean())  # Fill missing numeric values with the mean
    
    # Combine the numeric and non-numeric data back together
    df_cleaned = pd.concat([df_numeric_cleaned, df_non_numeric], axis=1)
    
    return df_cleaned
# Funktion 2: Feature Engineering
def feature_engineering(df):
    """
    Apply feature engineering to the dataset. This can involve creating new features.
    
    Parameters:
    df (DataFrame): Input pandas DataFrame.
    
    Returns:
    DataFrame: DataFrame with engineered features.
    """
    # Beispiel: Eine neue Feature-Spalte erstellen
    df['new_feature'] = df['existing_feature'] ** 2
    return df

# Funktion 3: Daten aufteilen
def split_data(df, target_column):
    """
    Split the data into training and testing sets.
    
    Parameters:
    df (DataFrame): Input pandas DataFrame.
    target_column (str): The name of the target column.
    
    Returns:
    tuple: Training and testing datasets (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Funktion 4: Modelltraining
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a model (RandomForest or GradientBoosting) on the training data.
    
    Parameters:
    X_train (DataFrame): Features for training.
    y_train (Series): Target values for training.
    model_type (str): Type of model to train ('random_forest' or 'gradient_boosting').
    
    Returns:
    model: Trained model.
    """
    if model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor()
    else:
        raise ValueError("Model type not supported. Choose 'random_forest' or 'gradient_boosting'.")
    
    model.fit(X_train, y_train)
    return model

# Funktion 5: Modellbewertung
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using the testing data and print the performance metrics.
    
    Parameters:
    model: Trained model to be evaluated.
    X_test (DataFrame): Features for testing.
    y_test (Series): True target values for testing.
    
    Returns:
    None
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

# Funktion 6: Plot der Vorhersagen
def plot_predictions(y_test, y_pred):
    """
    Plot the true vs predicted values.
    
    Parameters:
    y_test (Series): True target values.
    y_pred (Series): Predicted target values.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.show()

# Funktion 7: Polynomial Feature Transformation
def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to the dataset.
    
    Parameters:
    X (DataFrame): Input feature DataFrame.
    degree (int): Degree of the polynomial features.
    
    Returns:
    DataFrame: Transformed DataFrame with polynomial features.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly
