# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the dataset
# Ensure the dataset is placed in the correct path
df = pd.read_csv('dataset/Housing.csv')

# Step 2: Data Inspection
# Checking for missing values and data types to understand the dataset better
print("Missing values in each column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Step 3: Feature Engineering
# 3.1 Creating interaction terms (e.g., area * bedrooms)
df['area_bedrooms'] = df['area'] * df['bedrooms']

# 3.2 Log transformation of the target variable (price)
df['log_price'] = np.log(df['price'])

# Step 4: One-Hot Encoding for categorical variables
# Converting categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 5: Split the dataset into features (X) and target variable (y)
X = df.drop(['price', 'log_price'], axis=1)  # Features
y = df['log_price']  # Target variable

# Step 6: Handle Outliers (optional)
# Visualize outliers in the price variable using a boxplot
sns.boxplot(x=df['price'])
plt.show()

# Removing outliers based on a quantile threshold
# (Choosing to remove the top 5% of data points with extremely high prices)
threshold = df['price'].quantile(0.95)
df = df[df['price'] < threshold]

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (important for models like SVR)
# Scaling the feature values to ensure that all features contribute equally to the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Model Selection
# Trying different models: Random Forest, Linear Regression, Support Vector Regression (SVR)

# Initialize a dictionary of models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(kernel='linear')
}

# Random Forest with Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
rf_grid.fit(X_train_scaled, y_train)

# Step 10: Model Evaluation
# Predict the housing prices on the test set
y_pred = rf_grid.predict(X_test_scaled)

# Reverse the log transformation to get the actual price predictions
y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)

# Compare the actual prices vs. predicted prices
predictions_df = pd.DataFrame({
    'Actual Price': y_test_exp,
    'Predicted Price': y_pred_exp
})

# Print the first few rows to visualize
print("\nActual vs Predicted Prices:\n", predictions_df.head())

# Calculate and display evaluation metrics
mse = mean_squared_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_exp, y_pred_exp)

print(f'\nMean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')

# Step 11: Cross-Validation
# Perform cross-validation to ensure the model generalizes well
cv_scores = cross_val_score(rf_grid.best_estimator_, X, y, cv=5)
print(f'\nCross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

# Step 12: Feature Importance Analysis
# Display the most important features for the Random Forest model
importances = rf_grid.best_estimator_.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print the top 10 most important features
print("\nTop 10 Important Features:\n", feature_importance_df.head(10))

# Visualize the top 10 important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()

# Step 13: Save the best model for future use
# Save the trained Random Forest model and the scaler for future predictions
joblib.dump(rf_grid.best_estimator_, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Optionally, save actual vs predicted prices to a CSV file
predictions_df.to_csv('predicted_vs_actual.csv', index=False)
