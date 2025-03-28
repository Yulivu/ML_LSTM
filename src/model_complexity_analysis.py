"""
模型复杂度对模型表现的影响：测试不同的LSTM模型配置以找到最佳模型架构
"""
# %% 0.导入必要的库
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

# %% 1. 数据加载

print('加载数据...')
data = pd.read_csv(PROCESSED_DATA_DIR / 'sp500_features.csv')

# 转换日期格式，并按照日期排序
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# 提取目标列
target = 'Close'
# 提取除目标列和日期列之外的所有特征列
features = data.drop(columns=[target, 'Date']).columns.tolist()

print(f"使用的特征: {features}")
print(f"目标变量: {target}")

# 选取特征和目标
X = data[features].values
y = data[target].values.reshape(-1, 1)

# 数据归一化
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
def moving_average_smoothing(data, window_size=3):
    """对多维时间序列数据中的每个特征分别进行平滑处理"""
    smoothed_data = np.empty_like(data)
    for col in range(data.shape[1]):
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
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        for _ in range(num_augmentations):
            X_smooth = moving_average_smoothing(X[i])
            X_noisy = random_noise(X_smooth)
            X_shifted = time_series_shift(X_noisy)
            X_augmented.append(X_shifted)
            y_augmented.append(y[i])
    return np.array(X_augmented), np.array(y_augmented)


# 数据集划分
X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)

print(f"原始训练集形状: {X_train_full.shape}")
print(f"增强后训练集形状: {X_train_full_augmented.shape}")


# %% 3. 定义不同复杂度的 LSTM 模型
def build_model(lstm_layers=1, units=50, dense_layers=1):
    """
    构建不同复杂度的LSTM模型
    参数:
        lstm_layers: LSTM层的数量
        units: 每层LSTM的单元数
        dense_layers: Dense层的数量
    返回:
        model: 构建的LSTM模型
    """
    model = Sequential()
    model.add(LSTM(units, return_sequences=(lstm_layers > 1),
                   input_shape=(X_train_full_augmented.shape[1], X_train_full_augmented.shape[2]),
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    for i in range(1, lstm_layers):
        model.add(LSTM(units, return_sequences=(i < lstm_layers - 1), kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
    for _ in range(dense_layers):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(1))
    return model


# %% 4.不同复杂度的模型训练与验证集评估

print('测试不同模型复杂度...')
lstm_layer_options = [1, 2, 3]  # LSTM层数
unit_options = [50, 100, 200]  # 神经元数量
dense_layer_options = [1, 2, 3]  # Dense层数
results = {}
validation_results = []
best_model = None
best_val_mse = float('inf')
tscv = TimeSeriesSplit(n_splits=5)

# 存储模型配置和结果
model_configs = []

for lstm_layers in lstm_layer_options:
    for units in unit_options:
        for dense_layers in dense_layer_options:
            model_name = f"LSTM-{lstm_layers}_Units-{units}_Dense-{dense_layers}"
            print(f"训练模型: {model_name}")

            val_mse_list, val_rmse_list, val_mae_list, val_r2_list = [], [], [], []

            for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
                print(f"  折叠 {fold + 1}")
                X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
                y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]

                model = build_model(lstm_layers=lstm_layers, units=units, dense_layers=dense_layers)
                model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                          callbacks=[early_stopping], verbose=0)

                val_predictions = model.predict(X_val)
                val_predicted_prices = scaler_y.inverse_transform(val_predictions)
                val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

                val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
                val_rmse = np.sqrt(val_mse)
                val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
                val_r2 = r2_score(val_real_prices, val_predicted_prices)

                val_mse_list.append(val_mse)
                val_rmse_list.append(val_rmse)
                val_mae_list.append(val_mae)
                val_r2_list.append(val_r2)

            # 记录每个模型复杂度的验证集平均值
            avg_val_mse = np.mean(val_mse_list)
            avg_val_rmse = np.mean(val_rmse_list)
            avg_val_mae = np.mean(val_mae_list)
            avg_val_r2 = np.mean(val_r2_list)

            # 添加模型配置和性能到列表
            model_configs.append({
                'model_name': model_name,
                'lstm_layers': lstm_layers,
                'units': units,
                'dense_layers': dense_layers,
                'mse': avg_val_mse,
                'rmse': avg_val_rmse,
                'mae': avg_val_mae,
                'r2': avg_val_r2
            })

            validation_results.append({
                'Model': model_name,
                'MSE': avg_val_mse,
                'RMSE': avg_val_rmse,
                'MAE': avg_val_mae,
                'R²': avg_val_r2
            })

            # 更新最佳模型
            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                best_model = model_name

# 创建DataFrame并存储模型配置和性能
model_config_df = pd.DataFrame(model_configs)
model_config_df.to_csv(RESULTS_DIR / 'model_complexity_configs.csv', index=False)

print(f"最佳模型 (基于验证集): {best_model}")

# 保存验证集结果为 CSV
df_val_results = pd.DataFrame(validation_results)
df_val_results.to_csv(RESULTS_DIR / 'models_validation_results.csv', index=False)

# %% 5. 绘制验证集的性能条形对比图

# 从结果中提取各个指标
models = [result['Model'] for result in validation_results]
mse_values = [result['MSE'] for result in validation_results]
rmse_values = [result['RMSE'] for result in validation_results]
mae_values = [result['MAE'] for result in validation_results]
r2_values = [result['R²'] for result in validation_results]

# 设置对数纵坐标，并调整标签角度
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.2  # 设置条形宽度
index = np.arange(len(models))  # 模型复杂度的索引

# 绘制不同评价指标的条形图
plt.bar(index, mse_values, bar_width, label='MSE', color='blue')
plt.bar(index + bar_width, rmse_values, bar_width, label='RMSE', color='green')
plt.bar(index + 2 * bar_width, mae_values, bar_width, label='MAE', color='orange')
plt.bar(index + 3 * bar_width, r2_values, bar_width, label='R²', color='red')

# 设置对数纵坐标
plt.yscale('log')

# 添加标签、标题等
plt.xlabel('Model Complexity', fontsize=12)
plt.ylabel('Metric Values (Log Scale)', fontsize=12)
plt.title('Validation Set Evaluation Metrics for Different LSTM Model Complexities', fontsize=14)

# 调整横坐标标签的旋转角度
plt.xticks(index + bar_width, models, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'validation_metric_comparison.svg')
plt.close()

# %% 5. 使用最佳模型进行测试集预测
best_model_config = model_config_df.loc[model_config_df['mse'].idxmin()]
best_model_name = best_model_config['model_name']
lstm_layers = best_model_config['lstm_layers']
units = best_model_config['units']
dense_layers = best_model_config['dense_layers']

print(f"最佳模型配置: {best_model_name}")
print(f"LSTM层数: {lstm_layers}, 单元数: {units}, Dense层数: {dense_layers}")

# 使用最佳模型配置重新训练和评估
print("使用最佳模型配置进行训练...")
best_model = build_model(lstm_layers=lstm_layers, units=units, dense_layers=dense_layers)
best_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 使用一部分训练数据作为验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_augmented, y_train_full_augmented, test_size=0.2, shuffle=False
)

history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# 保存最佳模型
best_model.save(MODELS_DIR / f"{best_model_name}.h5")
print(f"最佳模型已保存: {MODELS_DIR / f'{best_model_name}.h5'}")

# 绘制训练损失和验证损失的变化
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'best_model_loss.svg')
plt.close()

# %% 6. 模型测试
print("模型预测...")
test_predictions = best_model.predict(X_test)
test_predicted_prices = scaler_y.inverse_transform(test_predictions)
test_real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

test_mse = mean_squared_error(test_real_prices, test_predicted_prices)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_real_prices, test_predicted_prices)
test_r2 = r2_score(test_real_prices, test_predicted_prices)

print(f"测试集 - MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, R²: {test_r2}")

# 保存测试集结果
results_df = pd.DataFrame({
    'Real_Price': test_real_prices.flatten(),
    'Predicted_Price': test_predicted_prices.flatten()
})
results_df.to_csv(RESULTS_DIR / f'{best_model_name}_test_results.csv', index=False)

# %% 7. 绘制测试集上的真实值和预测值
plt.figure(figsize=(10, 6))

# 提取对应的日期，确保与测试集预测的时间点匹配
dates = data['Date'].values[-len(test_real_prices):]

plt.plot(dates, test_real_prices, label="Real Prices", color='blue')
plt.plot(dates, test_predicted_prices, label="Predicted Prices", color='red',
         linestyle='--')
plt.title(f'{best_model_name}: Real vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'{best_model_name}_real_vs_predicted.svg')
plt.close()

# 保存最终性能指标
performance = {
    'Model': [best_model_name],
    'MSE': [test_mse],
    'RMSE': [test_rmse],
    'MAE': [test_mae],
    'R2': [test_r2]
}
pd.DataFrame(performance).to_csv(RESULTS_DIR / 'best_model_performance.csv', index=False)

print("模型复杂度分析完成！")