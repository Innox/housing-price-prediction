# Housing Price Prediction Model

This project aims to predict housing prices using a machine learning model. It utilizes Random Forest, Linear Regression, and Support Vector Regression (SVR) models to predict housing prices based on various features in the dataset.

## Project Overview
The dataset includes several housing features like:
- Area (square feet)
- Bedrooms
- Bathrooms
- Stories
- Main road access, guest room, basement, etc.

The model uses feature engineering, one-hot encoding, and model tuning to optimize prediction performance. The best performing model, a **Random Forest Regressor**, is saved for future predictions.

### Key Features:
- **Feature Engineering**: Interaction terms and log transformations.
- **Model Tuning**: GridSearchCV to fine-tune Random Forest hyperparameters.
- **Evaluation**: Performance metrics such as R², RMSE, and cross-validation scores.
- **Feature Importance**: Visualization of the top 10 important features.

## Prerequisites
Ensure you have Python installed. Then, install the required libraries by running:

```bash
pip install -r requirements.txt
