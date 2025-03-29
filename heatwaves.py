import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, jsonify

# Flask app setup
app = Flask(__name__)

# Load and preprocess multiple datasets
def load_data():
    folder_path = "datafiles"
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

    dataframes = []
    for file in all_files:
        try:
            df = pd.read_csv(file, low_memory=False)

            # Convert to strings and ensure correct formatting
            df[['YEAR', 'MN', 'DT']] = df[['YEAR', 'MN', 'DT']].astype(str).apply(lambda x: x.str.zfill(2))
            df['date'] = pd.to_datetime(df[['YEAR', 'MN', 'DT']].agg('-'.join, axis=1), format="%Y-%m-%d", errors='coerce')

            dataframes.append(df)
            print(f"Loaded: {file} ✅")
        except Exception as e:
            print(f"Error loading {file}: {e} ❌")

    if not dataframes:
        raise ValueError("No valid CSV files found!")

    combined_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()
    print(f"Data after merging: {combined_df.shape}")  # Debugging

    return combined_df

# Function to create sequences
def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Train model function
def train_model():
    df = load_data()  # Load data
    
    # Trim spaces from column names
    df.columns = df.columns.str.strip()

    # Ensure 'MAX' column exists
    if 'MAX' not in df.columns:
        raise ValueError("Column 'MAX' not found in dataset!")

    # Convert 'MAX' to numeric and handle NaNs
    df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')

    # Drop rows where 'MAX' is NaN
    df = df.dropna(subset=['MAX'])

    # Check if 'MAX' column is empty after cleaning
    if df.empty:
        raise ValueError("After cleaning, 'MAX' column has no valid data. Check input files!")

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['MAX'] = scaler.fit_transform(df[['MAX']])

    # Create sequences for training
    seq_length = 7
    X, y = create_sequences(df['MAX'].values, seq_length)

    # Reshape for CNN + LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data (90% train, 10% test)
    split = int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Build CNN + LSTM model
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save("heatwave_model.h5")
    print("Model training completed and saved successfully!")

    return model, scaler, df


# Predict next 7 days
def predict_next_7_days(model, scaler, df):
    seq_length = 7
    last_seq = df['MAX'].values[-seq_length:].reshape(1, seq_length, 1)
    
    predictions = []
    for _ in range(7):
        pred = model.predict(last_seq)[0][0]
        predictions.append(pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1, 0] = pred

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

# Heatwave detection (threshold 37°C)
def detect_heatwave(predictions):
    return [temp >= 37 for temp in predictions]

# Confidence interval (±1.5°C)
def calculate_confidence_interval(predictions, df):
    actual_values = df['MAX'].values[-7:]
    within_range = [(abs(pred - actual) <= 1.5) for pred, actual in zip(predictions, actual_values)]
    accuracy = sum(within_range) / len(within_range) * 100
    return accuracy

# Interactive visualization
def plot_predictions(predictions, df):
    actual_values = df['MAX'].values[-7:]

    plt.figure(figsize=(10, 5))
    plt.plot(range(7), actual_values, label="Actual", marker='o')
    plt.plot(range(7), predictions, label="Predicted", marker='s')
    plt.axhline(y=37, color='r', linestyle='--', label="Heatwave Threshold (37°C)")
    
    plt.legend()
    plt.xlabel("Days Ahead")
    plt.ylabel("Temperature (°C)")
    plt.title("7-Day Temperature Prediction")
    plt.show()

# Flask route for real-time visualization
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    predictions = predict_next_7_days(model, scaler, df)
    heatwave_flags = detect_heatwave(predictions)
    accuracy = calculate_confidence_interval(predictions, df)

    return jsonify({
        "predictions": predictions.tolist(),
        "heatwave_detected": heatwave_flags,
        "confidence_accuracy": accuracy
    })

if __name__ == "__main__":
    df = load_data()

    # Load or Train model
    model_path = "heatwave_model.h5"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['MAX'] = scaler.fit_transform(df[['MAX']])
    else:
        print("Training new model...")
        model, scaler, df = train_model()

    # Make predictions
    predictions = predict_next_7_days(model, scaler, df)
    heatwave_flags = detect_heatwave(predictions)
    accuracy = calculate_confidence_interval(predictions, df)

    print("\nPredictions:", predictions)
    print("Heatwave detected:", heatwave_flags)
    print(f"Confidence Accuracy: {accuracy:.2f}%")

    plot_predictions(predictions, df)

    app.run(debug=True)
