import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# Perform a log-transform on selected features (you can customize this)
log_transform_features = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
                          'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)',
                          'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

for feature in log_transform_features:
    X_train[feature] = np.log1p(X_train[feature])
    X_test[feature] = np.log1p(X_test[feature])

# Check for NaNs or infinities in the dataset after log transformation
if X.isnull().sum().sum() > 0 or not np.isfinite(X).all().all():
    print("There are still NaNs or infinities in the dataset after log transformation.")
else:
    # Check for zero variance features
    zero_variance_features = X.columns[X.var() == 0]
    if not zero_variance_features.empty:
        print("Removing zero variance features:", zero_variance_features.tolist())
        X.drop(zero_variance_features, axis=1, inplace=True)

# Perform collinearity analysis (VIF calculation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

# Print the VIF values to identify multicollinearity
print("VIF Values:")
print(vif)

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

# Print the results
print("Metrics for motor_updrs after log-transform and collinearity analysis:")
print("MAE:", mae_motor)
print("MSE:", mse_motor)
print("RMSE:", rmse_motor)
print("R-squared:", r2_motor)

print("\nMetrics for total_updrs after log-transform and collinearity analysis:")
print("MAE:", mae_total)
print("MSE:", mse_total)
print("RMSE:", rmse_total)
print("R-squared:", r2_total)
