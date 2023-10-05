import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the dataset
data = pd.read_csv("po2_data.csv")

# Select features and target variables
X = data[['age', 'sex', 'test_time', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
          'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
          'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
y_motor = data['motor_updrs']
y_total = data['total_updrs']
# Split the data into training and testing sets (60% training, 40% testing)
X_train, X_test, y_train_motor, y_test_motor, y_train_total, y_test_total = train_test_split(
    X, y_motor, y_total, test_size=0.4, random_state=42)
# Create and train Linear Regression models for motor_updrs and total_updrs
model_motor = LinearRegression()
model_motor.fit(X_train, y_train_motor)
model_total = LinearRegression()
model_total.fit(X_train, y_train_total)
# Make predictions
y_pred_motor = model_motor.predict(X_test)
y_pred_total = model_total.predict(X_test)

# Calculate metrics for motor_updrs
mae_motor = mean_absolute_error(y_test_motor, y_pred_motor)
mse_motor = mean_squared_error(y_test_motor, y_pred_motor)
rmse_motor = np.sqrt(mse_motor)
r2_motor = r2_score(y_test_motor, y_pred_motor)

# Calculate metrics for total_updrs
mae_total = mean_absolute_error(y_test_total, y_pred_total)
mse_total = mean_squared_error(y_test_total, y_pred_total)
rmse_total = np.sqrt(mse_total)
r2_total = r2_score(y_test_total, y_pred_total)

# Calculate adjusted R-squared for motor_updrs and total_updrs
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2_motor = 1 - ((1 - r2_motor) * (n - 1) / (n - p - 1))
adj_r2_total = 1 - ((1 - r2_total) * (n - 1) / (n - p - 1))

# Normalize RMSE by dividing it by the standard deviation of the target variable
std_motor = np.std(y_test_motor)
std_total = np.std(y_test_total)
nrmse_motor = rmse_motor / std_motor
nrmse_total = rmse_total / std_total
# Print the results
print("Metrics for motor_updrs:")
print("MAE:", mae_motor)
print("MSE:", mse_motor)
print("RMSE:", rmse_motor)
print("NRMSE:", nrmse_motor)
print("R-squared:", r2_motor)
print("Adjusted R-squared:", adj_r2_motor)

print("\nMetrics for total_updrs:")
print("MAE:", mae_total)
print("MSE:", mse_total)
print("RMSE:", rmse_total)
print("NRMSE:", nrmse_total)
print("R-squared:", r2_total)
print("Adjusted R-squared:", adj_r2_total)