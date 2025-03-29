import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# File paths (update if needed)
file_paths = [
    '43057_Table_3_Daily_NDCQ-2025-02-237.csv',
    '43003_Table_3_Daily_NDCQ-2025-02-237.csv'
]

# Load datasets with low_memory=False to suppress DtypeWarning
dataframes = [pd.read_csv(fp, low_memory=False) for fp in file_paths]
data = pd.concat(dataframes, ignore_index=True)

# Select relevant columns
features = ['RH', 'SLP', 'MSLP', 'DPT', 'FFF']
target = 'DBT'

# Ensure required columns exist
missing_columns = [col for col in features + [target] if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Convert selected columns to numeric (forcing errors to NaN)
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing values
data = data.dropna(subset=features + [target])

# Check if data is empty after cleaning
if data.empty:
    raise ValueError("No valid data available after cleaning. Check your dataset.")

# Print data summary to verify correctness
print("Data Summary:")
print(data[features + [target]].describe())

# Train-test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f}°C')

# Predict tomorrow's temperature
if not X.empty:
    latest_data = X.iloc[-1].values.reshape(1, -1)
    predicted_temp = model.predict(latest_data)[0]
    print(f'Predicted Temperature for Tomorrow: {predicted_temp:.2f}°C')

    # Agricultural Advice System
    def irrigation_advice(temp, humidity):
        if temp > 35:
            return "⚠️ High Temperature! Water crops frequently."
        elif temp < 10:
            return "❄️ Low Temperature! Reduce irrigation to avoid frost damage."
        elif humidity < 40:
            return "💧 Low Humidity! Increase irrigation."
        else:
            return "✅ Optimal Conditions! Follow regular irrigation schedule."

    advice = irrigation_advice(predicted_temp, latest_data[0][0])
    print(f'Irrigation Advice: {advice}')
else:
    print("⚠️ Not enough data to make predictions.")
