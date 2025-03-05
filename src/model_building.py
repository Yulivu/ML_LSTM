import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# 获取项目根目录和数据目录
current_file = Path(__file__)
ROOT_DIR = current_file.parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

# 确保目录存在
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#%% 1. 数据加载
print('加载数据...')
data = pd.read_csv(PROCESSED_DATA_DIR / 'sp500_features.csv')

# 转换日期格式，并按照日期排序
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# 提取目标列
target = 'Close'
# 提取除目标列之外的所有特征列
features = data.drop(columns=[target, 'Date']).columns.tolist()

print(f"使用的特征: {features}")
print(f"目标变量: {target}")

# 选取特征和目标
X = data[features].values
y = data[target].values.reshape(-1, 1)

# 数据归一化 (MinMaxScaler)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 创建输入序列和标签，使用滑动窗口
def create_sequences(X, y, time_steps=60):
    """
    创建时间序列的滑动窗口数据
    参数:
        X: 特征数据
        y: 目标数据
        time_steps: 时间窗口大小
    返回:
        Xs: 特征序列
        ys: 目标值
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

print(f"序列数据形状 - X: {X_lstm.shape}, y: {y_lstm.shape}")

#%% 2. 划分数据集
print('划分数据集...')
# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

#%% 3. 模型构建
# 构建LSTM模型
print('构建LSTM模型...')
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

print('训练LSTM模型...')
# 训练模型时，使用验证集评估
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# 保存模型
model.save(MODELS_DIR / 'base_lstm_model.h5')
print(f"模型已保存到: {MODELS_DIR / 'base_lstm_model.h5'}")

#%% 4. 预测（使用保留的测试集进行预测）
print('预测LSTM模型...')
predictions = model.predict(X_test)

# 反归一化
predicted_prices = scaler_y.inverse_transform(predictions)
real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 计算评价指标
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)

# 输出各个指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

#%% 5. 绘图
plt.figure(figsize=(8, 4))

# 提取对应的日期，确保与测试集预测的时间点匹配
dates = data['Date'].values[-len(real_prices):]

# 绘制真实股价
plt.plot(dates, real_prices, label="Real Prices", color='blue')

# 绘制预测股价
plt.plot(dates, predicted_prices, label="Predicted Prices", color='red', linestyle='--')

plt.title('Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'stock_prices.svg')
plt.close()

# 绘制训练损失和验证损失的变化
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'loss.svg')
plt.close()

# 使用 Pandas 保存预测结果到 CSV 文件
results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': real_prices.flatten(),
    'Predicted_Close': predicted_prices.flatten()
})
results_df.to_csv(RESULTS_DIR / 'predicted_stock_prices.csv', index=False)

print(f"结果已保存到: {RESULTS_DIR}")