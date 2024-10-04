# Water Level Prediction in Tunisian Dams

## Project Overview

This project focuses on predicting water levels in key Tunisian dams using various Machine Learning models. Water scarcity in Tunisia poses a significant challenge to agriculture, industry, and local communities. By leveraging historical weather data and dam water levels, the goal of this project is to create predictive models that assist in better water management and resource distribution.

## Table of Contents

1. [Project Overview](#Project-Overview)
2. [Data Collection](#Data-Collection)
3. [Data Cleaning and Feature Engineering](#Data-Cleaning-and-Feature-Engineering)
4. [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-EDA)
5. [Modeling](#Modeling)
6. [Results and Insights](#Results-and-Insights)
7. [Conclusion](#Conclusion)
8. [Tools and Libraries](#Tools-and-Libraries)

## Data Collection

### Datasets Used:
1. **Weather Data:** Including variables like temperature, rainfall, wind speed, and more.
2. **Dam Water Levels:** Historical water levels from key Tunisian dams.

These datasets span from 2014 to 2019, providing a detailed view of water storage changes over time.

## Data Cleaning and Feature Engineering

### Data Cleaning:
- Handled missing values using interpolation techniques.
- Removed outliers for consistent analysis.
- Aligned and formatted all datasets to ensure compatibility.

### Feature Engineering:
- Created Lag features (Lag 1, Lag 7, Lag 30) to capture the impact of rainfall from previous days.
- Added seasonal features like month and season to capture long-term weather patterns affecting dam levels.

## Exploratory Data Analysis (EDA)

The EDA revealed key patterns in the data:
- Strong correlations between rainfall and water levels.
- Seasonal trends where water levels are higher during rainy seasons.
- Weak correlations between temperature and water levels.

Visualizations, including heatmaps and scatterplots, highlighted these relationships and guided feature selection.

## Modeling

We experimented with three main models:
1. **Linear Regression:** Baseline model, performed poorly with high errors.
2. **Random Forest Regressor:** Achieved the best performance with an R² of 0.97 and low error rates.
3. **Gradient Boosting:** Slightly less accurate than Random Forest, but still a good performer.

Hyperparameter tuning was applied to improve model performance, with Random Forest showing the best results after optimization.

## Results and Insights

- **Random Forest Regressor** outperformed other models with an accuracy of 97%.
- Feature Importance analysis highlighted Lag 1, RMIL, and SILIANA as key contributors to water level predictions.
- SHAP values provided further insights into feature impact on the model's predictions.

## Conclusion

This project successfully developed a robust predictive model for water levels in Tunisian dams. The Random Forest model, with an R² of 0.97, demonstrated strong predictive performance, especially when using lag features and regional data. This model can assist local governments in managing water resources and implementing timely interventions in the face of water scarcity.

Future work can focus on:
- Integrating real-time data for continuous model updates.
- Exploring more advanced models like LSTM for time-series forecasting.

## Tools and Libraries

- **Python:** Core programming language used for data analysis and modeling.
- **Pandas:** Used for data manipulation and preprocessing.
- **Matplotlib/Seaborn:** Used for creating visualizations and exploring data patterns.
- **Scikit-learn:** Machine learning library for model training and evaluation.
- **SHAP:** Used for model explainability and feature importance analysis.

## How to Run the Project

1. Install the required libraries:  
   ```bash
   pip install pandas matplotlib seaborn scikit-learn shap
