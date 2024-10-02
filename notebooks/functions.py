import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function 1: Data Cleaning
import pandas as pd

# Function 1: Data Cleaning
def clean_data(df):
    """
    Clean the input dataset by filling missing values.
    """
    df_cleaned = df.fillna(df.mean())  # Fill missing values with mean
    return df_cleaned

# Function 2: Feature Engineering
def feature_engineering(df):
    """
    Create new features, such as time-based features, from the cleaned dataset.
    """
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' column is datetime
    df['half_of_year'] = df['date'].dt.month.apply(lambda x: 'First Half' if x <= 6 else 'Second Half')
    return df

# Add other functions as needed...


# Function 2: Feature Engineering
def feature_engineering(df):
    """
    Create new features including time-based features.
    Args:
    df (DataFrame): The cleaned dataframe.
    
    Returns:
    DataFrame: The dataframe with engineered features.
    """
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a new column 'half_of_year' to indicate the first half vs second half of the year
    df['half_of_year'] = df['date'].dt.month.apply(lambda x: 'First Half' if x <= 6 else 'Second Half')
    
    return df

# Function 3: Splitting Data
def split_data(df, target_column):
    """
    Split the dataframe into training and test sets.
    Args:
    df (DataFrame): The dataframe with features.
    target_column (str): The column to predict.
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function 4: Train Linear Regression Model
def train_model(X_train, y_train, model_type="linear"):
    """
    Train the model based on the type of model selected.
    Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    model_type (str): Type of model to train ('linear' or 'random_forest').
    
    Returns:
    model: The trained model.
    """
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Function 5: Evaluate the Model
def evaluate_model(y_test, y_pred):
    """
    Evaluate the model's performance.
    Args:
    y_test (Series): True values.
    y_pred (Series): Predicted values.
    
    Returns:
    None
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

# Function 6: Plot Actual vs Predicted Values
def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values.
    Args:
    y_test (Series): True values.
    y_pred (Series): Predicted values.
    
    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Water Levels')
    plt.ylabel('Predicted Water Levels')
    plt.title('Actual vs Predicted Water Levels')
    plt.show()

# Function 7: Plot Distribution of Features
def plot_distribution(df):
    """
    Plot the distributions of the features.
    Args:
    df (DataFrame): The dataframe with features.
    
    Returns:
    None
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['tavg'], kde=True)
    plt.title('Distribution of tavg')

    plt.subplot(1, 3, 2)
    sns.histplot(df['wspd'], kde=True)
    plt.title('Distribution of wspd')

    plt.subplot(1, 3, 3)
    sns.histplot(df['pres'], kde=True)
    plt.title('Distribution of pres')

    plt.tight_layout()
    plt.show()
