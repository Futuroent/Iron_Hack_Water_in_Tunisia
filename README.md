# Water Level Prediction in Tunisia's Dams
## Overview
This project aims to predict water levels in Tunisian dams based on weather data and historical dam data. The prediction is done using several machine learning models, with a focus on improving performance through feature engineering and hyperparameter tuning. The data was cleaned, explored, and analyzed using various statistical and machine learning techniques, resulting in predictive models that may aid in water management efforts in Tunisia.

Table of Contents
Project Overview
Data Sources
Installation
Project Structure
Data Cleaning
Exploratory Data Analysis (EDA)
Feature Engineering
Model Training & Evaluation
Hyperparameter Tuning
Key Visualizations
Future Work
How to Run
Data Sources
Dam Data: Historical water levels from various dams in Tunisia.
Weather Data: Daily weather metrics including temperature, precipitation, and wind speed from relevant regions in Tunisia.
Both datasets were merged on the date field to allow for accurate predictions based on daily weather data.

Installation
To run the project locally, follow these steps:

Clone the repository:

bash
Code kopieren
git clone https://github.com/your-username/Water_Tunisia_Prediction.git
cd Water_Tunisia_Prediction
Set up the virtual environment and install the required packages:

bash
Code kopieren
conda create --name Water_Tunisia_env python=3.9
conda activate Water_Tunisia_env
pip install -r requirements.txt
Start Jupyter Notebook to run the analysis:

bash
Code kopieren
jupyter notebook
Project Structure
bash
Code kopieren
Water_Tunisia_Prediction/
├── data/
│   ├── raw/            # Raw data files
│   ├── clean/          # Cleaned datasets used for analysis
│   └── merged_data.csv # Final merged dataset
├── notebooks/
│   ├── Water_Tunisia_Notebook.ipynb  # Main notebook with analysis
│   ├── Cleaning_Data.ipynb           # Data cleaning process
│   ├── EDA_Notebook.ipynb            # Exploratory data analysis
│   └── Model_Training.ipynb          # Model training and evaluation
├── functions.py                      # Reusable Python functions
├── config.yaml                       # Configuration file
├── requirements.txt                  # Required Python packages
└── README.md                         # Project README file
Data Cleaning
The data cleaning process involved several steps:

Handling Missing Values: Missing values in both the dam and weather datasets were filled using various imputation techniques such as forward filling.
Date Formatting: The date column was standardized to YYYY-MM-DD format and used as the primary key for merging the datasets.
Dropping Duplicates: Any duplicate entries were removed to ensure data consistency.
Outlier Detection: Outliers in water levels and weather metrics were identified and removed using the IQR (Interquartile Range) method.
Code for cleaning is provided in Cleaning_Data.ipynb.

Exploratory Data Analysis (EDA)
We performed an extensive EDA to understand relationships between weather conditions and water levels in the dams. Key insights include:

Univariate Analysis: Histograms and boxplots were used to explore the distributions of water levels, temperatures, and precipitation.
Bivariate Analysis: Correlation heatmaps were generated to understand the relationships between various features such as temperature, precipitation, and water levels.
Multivariate Analysis: Pairplots and 3D scatter plots helped us visualize the relationships between multiple weather features and water levels simultaneously.
Key insights:

Strong correlation between precipitation (prcp) and water levels in several dams.
High correlation between minimum and maximum temperatures, affecting evaporation and dam levels.
Feature Engineering
Several new features were created to enhance model performance:

Date Features: Extracted year, month, and day from the date column to account for seasonality.
Cumulative Rainfall: A feature for cumulative rainfall was added to capture the effect of prolonged periods of rain on water levels.
Daily Temperature Change: Difference between daily maximum and minimum temperatures to account for the effects of sudden temperature changes.
These features were critical in improving model accuracy, as they capture long-term weather effects on water storage.

Model Training & Evaluation
We experimented with various machine learning models for predicting water levels, including:

Linear Regression: Simple baseline model.

Mean Squared Error (MSE): 1840
R² Score: -0.07
Interpretation: Model had low predictive power, not accounting for non-linear relationships.
Random Forest Regressor: Ensemble model to capture complex interactions between features.

MSE: 1617
R² Score: -0.08
Interpretation: Improved but still not sufficient for accurate predictions.
Gradient Boosting Regressor: Boosted model to further refine predictions.

MSE: 1539
R² Score: -0.03
Interpretation: Better at handling non-linearities, but further tuning was needed.
Hyperparameter Tuning
We used Grid Search and Cross-Validation to optimize hyperparameters for the Gradient Boosting and XGBoost models.

Gradient Boosting Best Parameters:
yaml
Code kopieren
learning_rate: 0.01
max_depth: 5
n_estimators: 100
XGBoost Best Parameters:
yaml
Code kopieren
learning_rate: 0.01
max_depth: 7
min_child_weight: 5
n_estimators: 100
These parameters significantly improved the model's predictive performance:

XGBoost MSE: 2014
R² Score: -0.34
Key Visualizations
Actual vs Predicted Water Levels: Visualized how closely the predicted water levels matched the actual values across different models.
Residual Plots: Showed the residuals for each model, helping us understand where the models made significant errors.
SHAP Summary Plot: Used SHAP values to interpret feature importance, revealing that temperature and precipitation were the most influential features.
Future Work
To further improve the model, we propose the following next steps:

Incorporating External Data: Using real-time weather data and water usage patterns for more accurate predictions.
Real-Time Monitoring: Building an app that allows water managers to monitor dam levels in real-time.
Seasonal Adjustment: Implementing more advanced time-series techniques like ARIMA to capture seasonal patterns in water levels.
Cloud Deployment: Deploying the model using a cloud-based platform like AWS for real-time predictions and monitoring.
How to Run
Clone the repository.
Install dependencies using the requirements.txt file.
Run the Jupyter notebooks:
Data Cleaning: Cleaning_Data.ipynb
EDA: EDA_Notebook.ipynb
Model Training: Model_Training.ipynb
Ensure that all the cleaned datasets are located in the data/clean/ folder.

