import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Rest of your imports
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Display first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png')  # Save plot as image

# Splitting the data into features (X) and target (y)
X = data.drop(columns=['medv'])  # Drop the target column (median value)
y = data['medv']  # Target column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # Train the model

# Predictions using Linear Regression
y_pred_lr = lr_model.predict(X_test)

# Evaluation Metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Performance:")
print(f"Mean Squared Error: {mse_lr}")
print(f"R2 Score: {r2_lr}")

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Train the model

# Save the trained Random Forest model
joblib.dump(rf_model, 'rf_model.pkl')
print("Model saved as 'rf_model.pkl'")

# Predictions using Random Forest Regressor
y_pred_rf = rf_model.predict(X_test)

# Evaluation Metrics for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Performance:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R2 Score: {r2_rf}")

# Visualizing Predictions vs Actual Values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.7)
plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig('actual_vs_predicted.png')  # Save plot as an image
print("\nPrediction visualization saved as 'actual_vs_predicted.png'")
