"""
模型的改进：在基础LSTM模型上应用更多优化技术提高预测性能
"""
#%% 0.导入必要的库
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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

# %% 2. 数据增强与划分数据集
print('进行数据增强...')

# 使用数据增强方法
def moving_average_smoothing(data, window_size=3):
    """对多维时间序列数据中的每个特征分别进行平滑处理"""
    smoothed_data = np.empty_like(data)  # 创建与输入数据相同维度的空数组
    for col in range(data.shape[1]):  # 对每一列（每个特征）进行平滑处理
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data


def random_noise(data, noise_factor=0.01):
    """向时间序列数据添加随机噪声"""
    noise = noise_factor * np.random.randn(*data.shape)
    return data + noise


def time_series_shift(data, max_shift=5):
    """在时间序列中随机移动数据点"""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(data, shift, axis=0)


def data_augmentation(X, y, num_augmentations=5):
    """进行时间序列数据增强"""
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        # 原始数据
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        # 随机生成数据增强
        for _ in range(num_augmentations):
            # 增强方法1: 平滑处理
            X_smooth = moving_average_smoothing(X[i])
            # 增强方法2: 添加噪声
            X_noisy = random_noise(X_smooth)
            # 增强方法3: 时间偏移
            X_shifted = time_series_shift(X_noisy)

            # 保存增强数据
            X_augmented.append(X_shifted)
            y_augmented.append(y[i])  # 标签不变

    return np.array(X_augmented), np.array(y_augmented)


# 首先，将原始数据划分为训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

# 对训练集进行数据增强
X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)

print(f"原始训练数据形状: {X_train_full.shape}")
print(f"增强后训练数据形状: {X_train_full_augmented.shape}")

# %% 3.交叉验证
print('进行交叉验证...')

# 使用增强后的数据进行交叉验证
tscv = TimeSeriesSplit(n_splits=5)

mse_scores = []
rmse_scores = []
mae_scores = []
r2_scores = []

# 存储每个折叠的验证集预测结果
fold_predictions = []

for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
    print(f"折叠 {fold + 1}")

    # 训练集和验证集划分
    X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
    y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]

    # 构建改进的模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                   kernel_regularizer=l2(0.001)))  # 添加 L2 正则化
    model.add(Dropout(0.3))  # 增大 Dropout 率
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))  # 增大 Dropout 率
    model.add(Dense(units=1))

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

    # 训练模型时，使用验证集评估
    # 早停
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 在模型训练时使用 Early Stopping
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=1)

    # 分别绘制每个fold的训练和验证损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label=f'Fold {fold + 1} - Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label=f'Fold {fold + 1} - Validation Loss', color='orange', linestyle='--',
             linewidth=2)
    plt.title(f'Fold {fold + 1} - Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'loss_fold_{fold + 1}.svg')
    plt.close()

    # 在验证集上进行预测
    val_predictions = model.predict(X_val)

    # 反归一化预测值和真实值
    val_predicted_prices = scaler_y.inverse_transform(val_predictions)
    val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    # 存储预测结果
    fold_predictions.append({
        'fold': fold + 1,
        'real': val_real_prices.flatten(),
        'predicted': val_predicted_prices.flatten()
    })

    # 计算评价指标
    val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
    val_r2 = r2_score(val_real_prices, val_predicted_prices)

    # 记录每个fold的结果
    mse_scores.append(val_mse)
    rmse_scores.append(val_rmse)
    mae_scores.append(val_mae)
    r2_scores.append(val_r2)

    print(f"  Fold {fold + 1} - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

# 输出交叉验证的平均结果
print(f"交叉验证平均结果:")
print(f"Mean MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# 保存交叉验证结果
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'MSE': mse_scores,
    'RMSE': rmse_scores,
    'MAE': mae_scores,
    'R2': r2_scores
})
cv_results.to_csv(RESULTS_DIR / 'cross_validation_results.csv', index=False)

#%% 4. 训练最终模型
print('训练最终改进模型...')

# 使用所有训练数据训练最终模型
final_model = Sequential()
final_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_full_augmented.shape[1], X_train_full_augmented.shape[2]),
               kernel_regularizer=l2(0.001)))
final_model.add(Dropout(0.3))
final_model.add(LSTM(units=50, return_sequences=False))
final_model.add(Dropout(0.3))
final_model.add(Dense(units=1))

# 编译模型
final_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

# 设置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 分割出一部分训练数据用于验证
X_train, X_val, y_train, y_val = train_test_split(X_train_full_augmented, y_train_full_augmented, test_size=0.2, shuffle=False)

# 训练最终模型
history = final_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# 保存最终模型
final_model.save(MODELS_DIR / 'improved_lstm_model.h5')
print(f"改进模型已保存到: {MODELS_DIR / 'improved_lstm_model.h5'}")

# 绘制训练历史
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'improved_model_training_history.svg')
plt.close()

#%% 5. 预测（使用保留的测试集进行预测）
print('使用改进模型进行预测...')

# 预测验证集和测试集
test_predictions = final_model.predict(X_test)

# 反归一化预测值和真实值
test_predicted_prices = scaler_y.inverse_transform(test_predictions)
test_real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 计算测试集的评价指标
test_mse = mean_squared_error(test_real_prices, test_predicted_prices)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_real_prices, test_predicted_prices)
test_r2 = r2_score(test_real_prices, test_predicted_prices)

print(f"测试集结果:")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"R² Score: {test_r2:.4f}")

# 保存测试集结果
test_results = pd.DataFrame({
    'Model': ['Improved LSTM'],
    'MSE': [test_mse],
    'RMSE': [test_rmse],
    'MAE': [test_mae],
    'R2': [test_r2]
})
test_results.to_csv(RESULTS_DIR / 'improved_model_test_results.csv', index=False)

#%% 6. 绘图
plt.figure(figsize=(10, 6))

# 提取对应的日期，确保与测试集预测的时间点匹配
dates = data['Date'].values[-len(test_real_prices):]

# 绘制真实股价
plt.plot(dates, test_real_prices, label="Real Prices", color='blue')

# 绘制预测股价
plt.plot(dates, test_predicted_prices, label="Predicted Prices", color='red',
         linestyle='--')

plt.title('Improved Model: Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'improved_stock_prices_prediction.svg')
plt.close()

# 使用 Pandas 保存预测结果到 CSV 文件
results_df = pd.DataFrame({
    'Date': dates,
    'Real_Close': test_real_prices.flatten(),
    'Predicted_Close': test_predicted_prices.flatten(),
    'Error': np.abs(test_real_prices.flatten() - test_predicted_prices.flatten())
})
results_df.to_csv(RESULTS_DIR / 'improved_predicted_stock_prices.csv', index=False)

print(f"模型改进完成，所有结果保存到: {RESULTS_DIR}")