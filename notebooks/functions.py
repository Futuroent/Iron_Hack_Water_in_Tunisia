import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# This is your main/clean_data function
def main(df):
    # Data cleaning steps
    df_cleaned = df.dropna()  # Example cleaning step, adjust as needed
    return df_cleaned

def feature_engineering(df):
    # Perform feature engineering (e.g., creating polynomial features)
    df['average_water_level'] = df[['MELLEGUE', 'BEN METIR', 'KASSEB', 'BARBARA']].mean(axis=1)
    return df

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

def plot_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predictions')
    plt.show()

def plot_distribution(df, column):
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
