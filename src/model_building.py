"""
Dual Layer LSTM Model: Building and training a dual-layer LSTM model for stock price prediction
Added trend direction prediction evaluation
"""
# %% 0. Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get project root directory and data directory
current_file = Path(__file__)
ROOT_DIR = current_file.parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. Data Loading
print('Loading data...')
data = pd.read_csv(PROCESSED_DATA_DIR / 'sp500_features.csv')

# Convert date format and sort by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Extract target column
target = 'Close'
# Extract all feature columns except target
features = data.drop(columns=[target, 'Date']).columns.tolist()

print(f"Features used: {features}")
print(f"Target variable: {target}")

# Select features and target
X = data[features].values
y = data[target].values.reshape(-1, 1)

# Data normalization (MinMaxScaler)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


# Create input sequences and labels using sliding window
def create_sequences(X, y, time_steps=60):
    """
    Create sliding window data for time series
    Parameters:
        X: Feature data
        y: Target data
        time_steps: Window size
    Returns:
        Xs: Feature sequences
        ys: Target values
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 60
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

print(f"Sequence data shape - X: {X_lstm.shape}, y: {y_lstm.shape}")

# %% 2. Split dataset
print('Splitting dataset...')
# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# %% 3. Model Building
# Build dual layer LSTM model
print('Building dual layer LSTM model...')
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=200, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# Add early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print('Training dual layer LSTM model...')
# Train model using validation set for evaluation
history = model.fit(
    X_train, y_train,
    epochs=20,  # Set to 20 epochs, let early stopping decide
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Save model
model.save(MODELS_DIR / 'dual_layer_lstm_model.h5')
print(f"Model saved to: {MODELS_DIR / 'dual_layer_lstm_model.h5'}")

# %% 4. Prediction (using held-out test set)
print('Predicting with dual layer LSTM model...')
predictions = model.predict(X_test)

# Inverse transformation
predicted_prices = scaler_y.inverse_transform(predictions)
real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)

# Output metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")


# %% 5. Trend Direction Prediction Evaluation
def add_trend_evaluation(real_prices, predicted_prices):
    """
    Evaluate the model's ability to predict stock price trends
    """
    # Ensure input is one-dimensional array
    if real_prices.ndim > 1:
        real_prices = real_prices.flatten()
    if predicted_prices.ndim > 1:
        predicted_prices = predicted_prices.flatten()

    # Calculate actual and predicted trend directions (up=1, down=-1, unchanged=0)
    actual_trend = np.zeros(len(real_prices) - 1)
    predicted_trend = np.zeros(len(predicted_prices) - 1)

    for i in range(len(real_prices) - 1):
        # Actual trend
        if real_prices[i + 1] > real_prices[i]:
            actual_trend[i] = 1  # Up
        elif real_prices[i + 1] < real_prices[i]:
            actual_trend[i] = -1  # Down

        # Predicted trend
        if predicted_prices[i + 1] > predicted_prices[i]:
            predicted_trend[i] = 1  # Up
        elif predicted_prices[i + 1] < predicted_prices[i]:
            predicted_trend[i] = -1  # Down

    # Calculate trend direction prediction metrics
    true_up = np.sum((actual_trend == 1) & (predicted_trend == 1))
    true_down = np.sum((actual_trend == -1) & (predicted_trend == -1))
    false_up = np.sum((actual_trend == -1) & (predicted_trend == 1))
    false_down = np.sum((actual_trend == 1) & (predicted_trend == -1))

    # Total correct predictions
    correct_predictions = true_up + true_down
    total_predictions = len(actual_trend)
    direction_accuracy = correct_predictions / total_predictions

    # Print metrics
    print("\nTrend Direction Prediction Metrics:")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print("\nTrend Direction Matrix:")
    print(f"True Up (Correctly predicted upward trend): {true_up}")
    print(f"True Down (Correctly predicted downward trend): {true_down}")
    print(f"False Up (Predicted up but actually down): {false_up}")
    print(f"False Down (Predicted down but actually up): {false_down}")

    # Create trend direction matrix DataFrame
    trend_matrix = pd.DataFrame(
        [[true_up, false_down],
         [false_up, true_down]],
        index=['Actual Up', 'Actual Down'],
        columns=['Predicted Up', 'Predicted Down']
    )
    print("\nTrend Direction Matrix:")
    print(trend_matrix)

    # Visualize trend direction matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(trend_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Trend Direction Prediction Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'trend_direction_matrix.png', dpi=300)

    return direction_accuracy, trend_matrix


# Call trend evaluation function
direction_accuracy, trend_matrix = add_trend_evaluation(real_prices, predicted_prices)

# Save trend metrics to CSV
trend_metrics_df = pd.DataFrame({
    'Metric': ['Direction Accuracy', 'True Up', 'True Down', 'False Up', 'False Down'],
    'Value': [
        direction_accuracy,
        trend_matrix.loc['Actual Up', 'Predicted Up'],
        trend_matrix.loc['Actual Down', 'Predicted Down'],
        trend_matrix.loc['Actual Down', 'Predicted Up'],
        trend_matrix.loc['Actual Up', 'Predicted Down']
    ]
})
trend_metrics_df.to_csv(RESULTS_DIR / 'trend_direction_metrics.csv', index=False)

# %% 6. Plotting
plt.figure(figsize=(12, 6))

# Extract corresponding dates to match test set prediction points
dates = data['Date'].values[-len(real_prices):]

# Plot real stock prices - using solid blue line
plt.plot(dates, real_prices, label="Real Prices", color='blue', linestyle='-')

# Plot predicted stock prices - using solid red line
plt.plot(dates, predicted_prices, label="Predicted Prices", color='red', linestyle='-')

plt.title('Dual Layer LSTM: Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'stock_prices.png', dpi=300)
plt.close()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Dual Layer LSTM: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'loss.png', dpi=300)
plt.close()

# Plot training and validation MAE
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Dual Layer LSTM: Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'mae.png', dpi=300)
plt.close()

# Save prediction results to CSV file
results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': real_prices.flatten(),
    'Predicted_Close': predicted_prices.flatten()
})
results_df.to_csv(RESULTS_DIR / 'predicted_stock_prices.csv', index=False)

# Plot residuals analysis
plt.figure(figsize=(12, 6))
residuals = real_prices.flatten() - predicted_prices.flatten()
plt.plot(dates, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Dual Layer LSTM: Prediction Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals (Real - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'residuals.png', dpi=300)
plt.close()

# Plot residuals histogram
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
plt.title('Dual Layer LSTM: Residuals Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'residuals_hist.png', dpi=300)
plt.close()

print(f"Dual Layer LSTM model results saved to: {RESULTS_DIR}")