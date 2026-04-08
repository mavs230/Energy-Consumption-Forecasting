import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

#Loading dataset
df = pd.read_csv("energy_consumption.csv")

# Strip hidden spaces from column names to prevent KeyErrors
df.columns = df.columns.str.strip()

# Use One-Hot Encoding for categorical variables to avoid implying a false numerical order
df = pd.get_dummies(df, columns=['Building Type', 'Day of Week'], drop_first=True)

#Defining features
x = df.drop('Energy Consumption', axis=1)
y = df['Energy Consumption']
#Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Feature scaling (Fit on training data only to prevent data leakage)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#train linear regression model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

#predict on test set
y_pred = model.predict(x_test_scaled)

#Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 score: {r2:.4f}")
print(f"MAE: {mae:.2f} kWh")

# Feature Importance (Coefficients)
print("\n--- Model Coefficients ---")
print(f"Model Intercept: {model.intercept_:.2f}")
coefficients = pd.DataFrame({
    'Feature': x.columns,
    'Weight': model.coef_
}).sort_values(by='Weight', ascending=False)
print(coefficients)

#Interpretation
print("\nInterpretation")
print(f"R^2 = {r2:.4f} means {r2*100:.1f}% of the variance in energy consumption is explained by the model.")
print(f"MAE = {mae:.2f} kWh means on average, predictions are off by about {mae:.2f} kWh from actual consumption")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Energy Consumption (kWh)')
plt.ylabel('Predicted Energy Consumption (kWh)')
plt.title('Actual vs Predicted Energy Consumption (Perfect Linear Fit)')
plt.legend()
plt.show()