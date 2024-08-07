import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle

# Load your data (replace with your actual data path)
file_path = r'C:\Users\IPAT2024\Desktop\Divyansh\ReactorData\T-Model-Reactor-PSD+Offline-pH-Data.xlsx'
df = pd.read_excel(file_path)

# Drop rows with missing values in columns 'pH' and 'pH25'
df = df.dropna(subset=['pH', 'pH25'])

# Define features (X) and target (y)
X = df.drop(['pH25', 'pH', 'pH_Goal', 'PGV-X50'], axis=1)
X['pH'] = df['pH']
y = df['pH25']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit StandardScaler on X_train_poly (assuming all features are numeric)
scaler_poly = StandardScaler()
scaler_poly.fit(X_train_poly)

# Scale the polynomial features
X_train_poly_scaled = scaler_poly.transform(X_train_poly)
X_test_poly_scaled = scaler_poly.transform(X_test_poly)

# Train the Gradient Boosting Regressor model with polynomial features
gbr_poly = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_poly.fit(X_train_poly_scaled, y_train)

# Evaluate on training set
train_mse_poly = mean_squared_error(y_train, gbr_poly.predict(X_train_poly_scaled))
print(f'Training MSE (GBR with Polynomial Features): {train_mse_poly}')
print(f'Training R^2 (GBR with Polynomial Features): {gbr_poly.score(X_train_poly_scaled, y_train)}')

# Evaluate on testing set
test_mse_poly = mean_squared_error(y_test, gbr_poly.predict(X_test_poly_scaled))
print(f'Testing MSE (GBR with Polynomial Features): {test_mse_poly}')
print(f'Testing R^2 (GBR with Polynomial Features): {gbr_poly.score(X_test_poly_scaled, y_test)}')

# Plotting predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_train, gbr_poly.predict(X_train_poly_scaled), color='blue', label='Training Data')
plt.scatter(y_test, gbr_poly.predict(X_test_poly_scaled), color='red', label='Testing Data')
plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='black')
plt.title('Actual vs. Predicted (GBR with Polynomial Features)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# Save the trained GBR with polynomial features model using pickle
with open('gbr_poly_model_final.pkl', 'wb') as model_file:
    pickle.dump(gbr_poly, model_file)

# Save the scaler for polynomial features using pickle
with open('poly_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_poly, scaler_file)
