"""
窗口大小对模型表现的影响：测试不同的时间窗口大小对LSTM模型预测性能的影响
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
import seaborn as sns
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

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


# %% 2. 数据增强与划分数据集
def moving_average_smoothing(data, window_size=3):
    """对多维时间序列数据中的每个特征分别进行平滑处理"""
    smoothed_data = np.empty_like(data)  # 创建与输入数据相同维度的空数组
    for col in range(data.shape[1]):  # 对每一列（每个特征）进行平滑处理
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data


def data_augmentation(X, y, num_augmentations=1):
    """进行时间序列数据增强"""
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        # 原始数据
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        # 增强方法1: 平滑处理
        X_smooth = moving_average_smoothing(X[i])
        # 增强方法2: 添加噪声
        X_noisy = X_smooth + 0.01 * np.random.randn(*X_smooth.shape)
        # 增强方法3: 时间偏移
        X_shifted = np.roll(X_noisy, np.random.randint(-5, 5), axis=0)

        # 保存增强数据
        X_augmented.append(X_shifted)
        y_augmented.append(y[i])  # 标签不变

    return np.array(X_augmented), np.array(y_augmented)


# %% 3. 测试不同滑动窗口大小的效果
window_sizes = [10, 30, 60, 90, 120]  # 不同的滑动窗口大小
metrics = ['MSE', 'RMSE', 'MAE', 'R2']

# 存储不同窗口大小下的结果
results = {window_size: {metric: [] for metric in metrics} for window_size in window_sizes}

for window_size in window_sizes:
    print(f'分析窗口大小: {window_size}天')

    # 创建不同窗口大小的输入序列
    X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, window_size)

    # 将原始数据划分为训练集和测试集，保留测试集用于最终评估
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

    # 对训练集进行数据增强
    X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)

    # 使用 TimeSeriesSplit 进行交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
        print(f"窗口大小 {window_size}天 的第 {fold + 1} 折")

        # 划分训练集和验证集
        X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
        y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]

        # 构建模型
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2]),
                       kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))

        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=0)

        # 在验证集上进行预测
        val_predictions = model.predict(X_val)
        val_predicted_prices = scaler_y.inverse_transform(val_predictions)
        val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

        # 计算评价指标
        val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
        val_r2 = r2_score(val_real_prices, val_predicted_prices)

        # 记录当前窗口大小和 fold 的指标
        results[window_size]['MSE'].append(val_mse)
        results[window_size]['RMSE'].append(val_rmse)
        results[window_size]['MAE'].append(val_mae)
        results[window_size]['R2'].append(val_r2)

# 将结果转换为DataFrame并保存
results_data = []
for window_size in window_sizes:
    for metric in metrics:
        for fold_value in results[window_size][metric]:
            results_data.append({
                'window_size': window_size,
                'metric': metric,
                'value': fold_value
            })

results_df = pd.DataFrame(results_data)
results_df.to_csv(RESULTS_DIR / 'window_size_performance_raw.csv', index=False)

# %% 4. 计算每个窗口大小下的指标平均值，并准备可视化数据
avg_results = []
for window_size in window_sizes:
    avg_metrics = {}
    for metric in metrics:
        avg_metrics[metric] = np.mean(results[window_size][metric])
    avg_results.append({
        'window_size': window_size,
        **avg_metrics
    })

# 保存平均结果
avg_results_df = pd.DataFrame(avg_results)
avg_results_df.to_csv(RESULTS_DIR / 'window_size_performance_avg.csv', index=False)

# 对指标进行对数变换（仅对 MSE 和 RMSE 进行处理）
log_transformed_results = []
for row in avg_results:
    log_transformed_results.append({
        'window_size': row['window_size'],
        'MSE': np.log1p(row['MSE']),  # 对 MSE 进行对数变换
        'RMSE': np.log1p(row['RMSE']),  # 对 RMSE 进行对数变换
        'MAE': row['MAE'],  # 保持 MAE 不变
        'R2': row['R2']  # 保持 R² 不变
    })

log_results_df = pd.DataFrame(log_transformed_results)
log_results_df.to_csv(RESULTS_DIR / 'window_size_performance_log.csv', index=False)

# %% 5. 可视化不同窗口大小下的性能

# 创建适合多个图表的画布
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Effect of Window Size on Model Performance', fontsize=16)

# 平面化axs数组，便于遍历
axs = axs.flatten()

# 创建每个指标的专门图表
for i, metric in enumerate(metrics):
    # 使用原始数据（非对数变换）绘图
    metric_values = [row[metric] for row in avg_results]
    axs[i].plot(window_sizes, metric_values, marker='o', linestyle='-', color='blue', linewidth=2)
    axs[i].set_title(f'{metric} vs Window Size')
    axs[i].set_xlabel('Window Size (days)')
    axs[i].set_ylabel(metric)
    axs[i].grid(True)

    # 为每个点添加标签
    for x, y in zip(window_sizes, metric_values):
        axs[i].annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
plt.savefig(RESULTS_DIR / 'window_size_metrics.svg')
plt.close()

# %% 6. 使用条形图比较不同窗口大小下的性能
# 设置条形图
barWidth = 0.15  # 增大条形图的宽度
r = np.arange(len(metrics))  # 设置基础横坐标位置
plt.figure(figsize=(12, 6))  # 调整图形大小

# 使用Seaborn调色板
colors = sns.color_palette("Blues", len(window_sizes))  # 使用更加美观的调色板

# 绘制不同窗口大小的指标
for idx, window_size in enumerate(window_sizes):
    avg_metrics = [log_transformed_results[idx][metric] for metric in metrics]
    bars = plt.bar(r + idx * barWidth, avg_metrics, width=barWidth, color=colors[idx],
                   label=f'Window Size {window_size}')

    # 在每个条形图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# 添加图例和标签
plt.xlabel('Metrics', fontweight='bold')
plt.ylabel('Log-Transformed / Original Value', fontweight='bold')

# 调整 xticks 的位置，使其位于每组条形图的中央
plt.xticks([r + barWidth * (len(window_sizes) / 2 - 0.5) for r in np.arange(len(metrics))], metrics)

plt.title('Log-Transformed Evaluation Metrics for Different Window Sizes')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'window_size_performance_bar.svg')
plt.close()

# %% 7. 找出最佳窗口大小并进行最终模型训练

# 基于MAE选择最佳窗口大小（较小的MAE表示更好的性能）
best_window_index = np.argmin([row['MAE'] for row in avg_results])
best_window_size = window_sizes[best_window_index]
print(f"最佳窗口大小: {best_window_size}天")

# 使用最佳窗口大小创建序列
X_lstm_best, y_lstm_best = create_sequences(X_scaled, y_scaled, best_window_size)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm_best, y_lstm_best, test_size=0.2, shuffle=False)
X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)

# 进一步划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_full_augmented, y_train_full_augmented, test_size=0.2,
                                                  shuffle=False)

# 使用最佳窗口大小构建最终模型
print(f"使用窗口大小 {best_window_size}天 训练最终模型...")
final_model = Sequential()
final_model.add(LSTM(units=50, return_sequences=True,
                     input_shape=(X_train.shape[1], X_train.shape[2]),
                     kernel_regularizer=l2(0.001)))
final_model.add(Dropout(0.3))
final_model.add(LSTM(units=50, return_sequences=False))
final_model.add(Dropout(0.3))
final_model.add(Dense(units=1))

final_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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
final_model.save(MODELS_DIR / f'lstm_window_{best_window_size}.h5')
print(f"最终模型已保存到: {MODELS_DIR / f'lstm_window_{best_window_size}.h5'}")

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
plt.savefig(RESULTS_DIR / f'training_history_window_{best_window_size}.svg')
plt.close()

# %% 8. 在测试集上评估最终模型
print("在测试集上评估最终模型...")
test_predictions = final_model.predict(X_test)
test_predicted_prices = scaler_y.inverse_transform(test_predictions)
test_real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 计算评价指标
test_mse = mean_squared_error(test_real_prices, test_predicted_prices)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_real_prices, test_predicted_prices)
test_r2 = r2_score(test_real_prices, test_predicted_prices)

print(f"测试集评估结果 (窗口大小 = {best_window_size}天):")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R²: {test_r2:.4f}")

# 保存测试结果
test_results = {
    'window_size': best_window_size,
    'MSE': test_mse,
    'RMSE': test_rmse,
    'MAE': test_mae,
    'R2': test_r2
}
pd.DataFrame([test_results]).to_csv(RESULTS_DIR / 'best_window_test_results.csv', index=False)

# 保存预测结果
test_dates = data['Date'].values[-len(test_real_prices):]
predictions_df = pd.DataFrame({
    'Date': test_dates,
    'Real_Price': test_real_prices.flatten(),
    'Predicted_Price': test_predicted_prices.flatten()
})
predictions_df.to_csv(RESULTS_DIR / f'predictions_window_{best_window_size}.csv', index=False)

# 绘制预测结果对比图
plt.figure(figsize=(12, 6))
plt.plot(test_dates, test_real_prices, label='Real Prices', color='blue')
plt.plot(test_dates, test_predicted_prices, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'Real vs Predicted Stock Prices (Window Size = {best_window_size} days)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'prediction_comparison_window_{best_window_size}.svg')
plt.close()

print("时间窗口大小分析完成！")