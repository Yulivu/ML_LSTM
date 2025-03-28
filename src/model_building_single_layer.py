"""
Complete LSTM Analysis:
1. Train single and dual layer LSTM models
2. Compare with lagged values
3. Create trend confusion matrices
4. Better match analysis
"""
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
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# 获取项目根目录和数据目录
# 导入项目路径配置
# 获取当前文件的路径
current_file = Path(__file__)
# 获取项目根目录
ROOT_DIR = current_file.parent.absolute()

# 定义常用目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXPLORATION_DIR = DATA_DIR / "exploration"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = EXPLORATION_DIR / "figures"  # 新增图片保存目录
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

# 确保所有必要的目录都存在
for directory in [DATA_DIR, RAW_DATA_DIR, EXPLORATION_DIR, PROCESSED_DATA_DIR, FIGURES_DIR,MODELS_DIR,RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== 1. LOAD AND PREPARE DATA =====
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
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 60
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

print(f"Sequence data shape - X: {X_lstm.shape}, y: {y_lstm.shape}")

# Split dataset
print('Splitting dataset...')
X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# ===== 2. TRAIN SINGLE LAYER LSTM MODEL =====
print('Building and training single layer LSTM model...')
single_model = Sequential()
single_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
single_model.add(Dropout(0.2))
single_model.add(Dense(units=1))

single_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
single_model.summary()

# Add early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train model
single_history = single_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Save model
single_model.save(MODELS_DIR / 'single_layer_lstm_model.h5')

# ===== 3. TRAIN DUAL LAYER LSTM MODEL =====
print('Building and training dual layer LSTM model...')
dual_model = Sequential()
dual_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
dual_model.add(Dropout(0.2))
dual_model.add(LSTM(units=50, return_sequences=False))
dual_model.add(Dropout(0.2))
dual_model.add(Dense(units=1))

dual_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
dual_model.summary()

# Train model
dual_history = dual_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Save model
dual_model.save(MODELS_DIR / 'dual_layer_lstm_model.h5')

# ===== 4. MAKE PREDICTIONS =====
print('Making predictions...')
single_pred = single_model.predict(X_test)
dual_pred = dual_model.predict(X_test)

# Inverse transform predictions
single_pred = scaler_y.inverse_transform(single_pred)
dual_pred = scaler_y.inverse_transform(dual_pred)
real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Create one-day lagged actual values
lagged_real_prices = np.roll(real_prices, 1)
lagged_real_prices[0] = lagged_real_prices[1]  # Fix the first value

# Get dates corresponding to test set
dates = data['Date'].values[-len(real_prices):]

# ===== 5. CALCULATE MODEL COMPARISON METRICS =====
print('Calculating metrics...')

# Single layer vs real
single_mse = mean_squared_error(real_prices, single_pred)
single_rmse = np.sqrt(single_mse)
single_mae = mean_absolute_error(real_prices, single_pred)
single_r2 = r2_score(real_prices, single_pred)

# Dual layer vs real
dual_mse = mean_squared_error(real_prices, dual_pred)
dual_rmse = np.sqrt(dual_mse)
dual_mae = mean_absolute_error(real_prices, dual_pred)
dual_r2 = r2_score(real_prices, dual_pred)

# Single layer vs lagged
single_lag_mse = mean_squared_error(lagged_real_prices, single_pred)
single_lag_rmse = np.sqrt(single_lag_mse)
single_lag_mae = mean_absolute_error(lagged_real_prices, single_pred)
single_lag_r2 = r2_score(lagged_real_prices, single_pred)

# Dual layer vs lagged
dual_lag_mse = mean_squared_error(lagged_real_prices, dual_pred)
dual_lag_rmse = np.sqrt(dual_lag_mse)
dual_lag_mae = mean_absolute_error(lagged_real_prices, dual_pred)
dual_lag_r2 = r2_score(lagged_real_prices, dual_pred)

# Print comparison metrics
print("\nModel vs Real Metrics:")
print(f"{'Metric':<15} {'Single Layer':<15} {'Dual Layer':<15}")
print(f"{'-' * 45}")
print(f"{'MSE':<15} {single_mse:<15.2f} {dual_mse:<15.2f}")
print(f"{'RMSE':<15} {single_rmse:<15.2f} {dual_rmse:<15.2f}")
print(f"{'MAE':<15} {single_mae:<15.2f} {dual_mae:<15.2f}")
print(f"{'R²':<15} {single_r2:<15.4f} {dual_r2:<15.4f}")

print("\nModel vs Lagged Metrics:")
print(f"{'Metric':<15} {'Single Layer':<15} {'Dual Layer':<15}")
print(f"{'-' * 45}")
print(f"{'MSE':<15} {single_lag_mse:<15.2f} {dual_lag_mse:<15.2f}")
print(f"{'RMSE':<15} {single_lag_rmse:<15.2f} {dual_lag_rmse:<15.2f}")
print(f"{'MAE':<15} {single_lag_mae:<15.2f} {dual_lag_mae:<15.2f}")
print(f"{'R²':<15} {single_lag_r2:<15.4f} {dual_lag_r2:<15.4f}")

# ===== 6. PLOT MODEL COMPARISON =====
print('Creating model comparison plot...')
plt.figure(figsize=(12, 6))
plt.plot(dates, real_prices, label="Real Prices", color='blue', linestyle='-')
plt.plot(dates, single_pred, label="Single Layer LSTM", color='red', linestyle='-')
plt.plot(dates, dual_pred, label="Dual Layer LSTM", color='green', linestyle='-')
plt.title('Model Comparison: Real vs Single Layer vs Dual Layer')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300)
plt.close()

# ===== 7. PLOT LAG COMPARISON =====
print('Creating lag comparison plot...')
plt.figure(figsize=(12, 6))
plt.plot(dates, real_prices, label="Real Prices", color='blue', linestyle='-')
plt.plot(dates, lagged_real_prices, label="One-Day Lagged Prices", color='purple', linestyle='-')
plt.plot(dates, single_pred, label="Single Layer LSTM", color='red', linestyle='-')
plt.plot(dates, dual_pred, label="Dual Layer LSTM", color='green', linestyle='-')
plt.title('Lag Comparison: Real vs Lagged vs Models')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lag_comparison.png', dpi=300)
plt.close()

# ===== 8. TREND DIRECTION ANALYSIS =====
print('Creating trend direction confusion matrices...')


def create_trend_matrix(real_prices, predicted_prices, name):
    """Create trend direction confusion matrix"""
    # Ensure inputs are one-dimensional arrays
    if real_prices.ndim > 1:
        real_prices = real_prices.flatten()
    if predicted_prices.ndim > 1:
        predicted_prices = predicted_prices.flatten()

    # Calculate trend directions (up=1, down=-1)
    real_trend = np.zeros(len(real_prices) - 1)
    pred_trend = np.zeros(len(predicted_prices) - 1)

    for i in range(len(real_prices) - 1):
        # Real trend
        if real_prices[i + 1] > real_prices[i]:
            real_trend[i] = 1  # Up
        elif real_prices[i + 1] < real_prices[i]:
            real_trend[i] = -1  # Down

        # Predicted trend
        if predicted_prices[i + 1] > predicted_prices[i]:
            pred_trend[i] = 1  # Up
        elif predicted_prices[i + 1] < predicted_prices[i]:
            pred_trend[i] = -1  # Down

    # Calculate confusion matrix values
    true_up = np.sum((real_trend == 1) & (pred_trend == 1))
    true_down = np.sum((real_trend == -1) & (pred_trend == -1))
    false_up = np.sum((real_trend == -1) & (pred_trend == 1))
    false_down = np.sum((real_trend == 1) & (pred_trend == -1))

    # Calculate accuracy
    total = len(real_trend)
    accuracy = (true_up + true_down) / total

    # Create trend matrix
    matrix = pd.DataFrame(
        [[true_up, false_down],
         [false_up, true_down]],
        index=['Actual Up', 'Actual Down'],
        columns=['Predicted Up', 'Predicted Down']
    )

    # Print results
    print(f"\n{name} Trend Direction Accuracy: {accuracy:.2%}")
    print(f"{name} Trend Direction Matrix:")
    print(matrix)

    # Visualize the matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Trend Direction Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name.lower().replace(" ", "_")}_trend_matrix.png', dpi=300)
    plt.close()

    return accuracy, matrix


# Create trend matrices for all four comparisons
single_real_acc, single_real_matrix = create_trend_matrix(real_prices, single_pred, "Single Layer vs Real")
dual_real_acc, dual_real_matrix = create_trend_matrix(real_prices, dual_pred, "Dual Layer vs Real")
single_lag_acc, single_lag_matrix = create_trend_matrix(lagged_real_prices, single_pred, "Single Layer vs Lagged")
dual_lag_acc, dual_lag_matrix = create_trend_matrix(lagged_real_prices, dual_pred, "Dual Layer vs Lagged")

# ===== 9. BETTER MATCH ANALYSIS =====
print('Performing better match analysis...')

# For single layer
single_real_diff = np.abs(real_prices.flatten() - single_pred.flatten())
single_lag_diff = np.abs(lagged_real_prices.flatten() - single_pred.flatten())
single_better_match = np.where(single_real_diff < single_lag_diff, 'Real', 'Lagged')
single_real_better = np.sum(single_better_match == 'Real')
single_lag_better = np.sum(single_better_match == 'Lagged')

# For dual layer
dual_real_diff = np.abs(real_prices.flatten() - dual_pred.flatten())
dual_lag_diff = np.abs(lagged_real_prices.flatten() - dual_pred.flatten())
dual_better_match = np.where(dual_real_diff < dual_lag_diff, 'Real', 'Lagged')
dual_real_better = np.sum(dual_better_match == 'Real')
dual_lag_better = np.sum(dual_better_match == 'Lagged')

# Print results
print("\nBetter Match Analysis:")
print(f"Single Layer - Days closer to real: {single_real_better} ({single_real_better / len(real_prices):.2%})")
print(f"Single Layer - Days closer to lagged: {single_lag_better} ({single_lag_better / len(real_prices):.2%})")
print(f"Dual Layer - Days closer to real: {dual_real_better} ({dual_real_better / len(real_prices):.2%})")
print(f"Dual Layer - Days closer to lagged: {dual_lag_better} ({dual_lag_better / len(real_prices):.2%})")

# Create pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Single layer pie chart
ax1.pie([single_real_better, single_lag_better],
        labels=['Closer to Real', 'Closer to Lagged'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        explode=(0.1, 0))
ax1.set_title('Single Layer LSTM: Better Match Analysis')

# Dual layer pie chart
ax2.pie([dual_real_better, dual_lag_better],
        labels=['Closer to Real', 'Closer to Lagged'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        explode=(0.1, 0))
ax2.set_title('Dual Layer LSTM: Better Match Analysis')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'better_match_analysis.png', dpi=300)
plt.close()

# ===== 10. SAVE RESULTS =====
print('Saving results...')

# Save model comparison metrics
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²',
               'MSE vs Lagged', 'RMSE vs Lagged', 'MAE vs Lagged', 'R² vs Lagged',
               'Trend Accuracy vs Real', 'Trend Accuracy vs Lagged',
               'Better Match - Real', 'Better Match - Lagged'],
    'Single Layer': [single_mse, single_rmse, single_mae, single_r2,
                     single_lag_mse, single_lag_rmse, single_lag_mae, single_lag_r2,
                     single_real_acc, single_lag_acc,
                     single_real_better / len(real_prices), single_lag_better / len(real_prices)],
    'Dual Layer': [dual_mse, dual_rmse, dual_mae, dual_r2,
                   dual_lag_mse, dual_lag_rmse, dual_lag_mae, dual_lag_r2,
                   dual_real_acc, dual_lag_acc,
                   dual_real_better / len(real_prices), dual_lag_better / len(real_prices)]
})
metrics_df.to_csv(RESULTS_DIR / 'model_comparison_metrics.csv', index=False)

# Save prediction results
results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': real_prices.flatten(),
    'Lagged_Real_Close': lagged_real_prices.flatten(),
    'Single_Layer_Predicted': single_pred.flatten(),
    'Dual_Layer_Predicted': dual_pred.flatten(),
    'Single_Real_Diff': single_real_diff,
    'Single_Lag_Diff': single_lag_diff,
    'Single_Better_Match': single_better_match,
    'Dual_Real_Diff': dual_real_diff,
    'Dual_Lag_Diff': dual_lag_diff,
    'Dual_Better_Match': dual_better_match
})
results_df.to_csv(RESULTS_DIR / 'prediction_results.csv', index=False)

# Save individual model results
single_results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': real_prices.flatten(),
    'Predicted_Close': single_pred.flatten()
})
single_results_df.to_csv(RESULTS_DIR / 'single_layer_predicted_stock_prices.csv', index=False)

dual_results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': real_prices.flatten(),
    'Predicted_Close': dual_pred.flatten()
})
dual_results_df.to_csv(RESULTS_DIR / 'predicted_stock_prices.csv', index=False)

print('Analysis completed successfully!')