"""
基于LSTM的股价预测: 特征工程
包括：特征提取和特征处理
"""
#%% 0. 导入必要的库
import pandas as pd
import pandas_datareader.data as web
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# 获取项目根目录和数据目录
current_file = Path(__file__)
ROOT_DIR = current_file.parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
EXPLORATION_DIR = DATA_DIR / 'exploration'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 确保目录存在
EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

#%% 1. 读取数据集
sp500_data = pd.read_csv(RAW_DATA_DIR / 'SPX.csv', parse_dates=['Date'])
sp500_data = sp500_data.set_index('Date').sort_index()
# 考虑 2000 年后的数据
sp500_data_filtered = sp500_data.loc['2000-01-01':]

#%% 2. 时间序列特征

# 时间序列分解（趋势和季节性）
decompose_result = seasonal_decompose(sp500_data_filtered['Close'], model='multiplicative', period=252)
sp500_data_filtered['Trend'] = decompose_result.trend
sp500_data_filtered['Seasonal'] = decompose_result.seasonal
sp500_data_filtered['Residual'] = decompose_result.resid

# 计算波动性（20天滚动标准差）
sp500_data_filtered['Volatility'] = sp500_data_filtered['Close'].rolling(window=20).std()

#%% 3. 交易量特征
# 交易量变化率
sp500_data_filtered['Volume_Change'] = sp500_data_filtered['Volume'].pct_change()

# 计算交易量的Z-score
sp500_data_filtered['Volume_Zscore'] = stats.zscore(sp500_data_filtered['Volume'])

# 设定阈值，Z-score超过3或小于-3的交易量被视为异常交易量
sp500_data_filtered['Volume_Anomaly_Zscore'] = (sp500_data_filtered['Volume_Zscore'].abs() > 3).astype(int)

#%% 4. 回报率与价格特征

# 计算对数收益率
sp500_data_filtered['Log_Return'] = np.log(sp500_data_filtered['Close'] / sp500_data_filtered['Close'].shift(1))

#%% 5. 宏观经济特征

# 设置数据起始时间
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2020, 11, 4)

# 获取联邦基金利率数据
interest_rate_data = web.DataReader('DFF', 'fred', start, end)

# 获取CPI数据
cpi_data = web.DataReader('CPIAUCSL', 'fred', start, end)
# 计算通胀率：CPI变化的百分比
cpi_data['Inflation_Rate'] = cpi_data.pct_change() * 100

# 重命名列，确保列名唯一，便于合并
interest_rate_data.columns = ['Interest_Rate']
cpi_data.columns = ['CPI', 'Inflation_Rate']

# 合并两个DataFrame
macro_data = pd.merge(interest_rate_data, cpi_data, left_index=True, right_index=True, how='outer')

# 按日期排序
macro_data.sort_index(inplace=True)

# 处理缺失值（使用插值或前向填充）
macro_data['CPI'] = macro_data['CPI'].interpolate(method='linear')   # 插值法
macro_data['Inflation_Rate'] = macro_data['Inflation_Rate'].ffill()  # 前向填充
macro_data['Interest_Rate'] = macro_data['Interest_Rate'].ffill()  # 前向填充

# 将宏观经济数据添加到股票数据中
sp500_data_filtered['CPI'] = macro_data['CPI'].reindex(sp500_data_filtered.index, method='ffill')
sp500_data_filtered['Inflation_Rate'] = macro_data['Inflation_Rate'].reindex(sp500_data_filtered.index, method='ffill')
sp500_data_filtered['Interest_Rate'] = macro_data['Interest_Rate'].reindex(sp500_data_filtered.index, method='ffill')

#%% 6. 异常值处理

# 交易量 IQR 异常值检测
Q1_volume = sp500_data_filtered['Volume'].quantile(0.25)
Q3_volume = sp500_data_filtered['Volume'].quantile(0.75)
IQR_volume = Q3_volume - Q1_volume

lower_bound_volume = Q1_volume - 1.5 * IQR_volume
upper_bound_volume = Q3_volume + 1.5 * IQR_volume

# 检测交易量中的异常值
sp500_data_filtered['Volume_outlier'] = ((sp500_data_filtered['Volume'] < lower_bound_volume) |
                                         (sp500_data_filtered['Volume'] > upper_bound_volume))

# 对异常值进行修复
sp500_data_filtered['Volume_corrected'] = sp500_data_filtered['Volume']
sp500_data_filtered.loc[sp500_data_filtered['Volume_outlier'], 'Volume_corrected'] = np.nan  # 将异常值设为 NaN
sp500_data_filtered['Volume_corrected'] = sp500_data_filtered['Volume_corrected'].interpolate()  # 使用插值法修复

# 修复前后的交易量对比绘图
plt.figure(figsize=(8, 6))

# 子图1: 修复前的交易量
plt.subplot(2, 1, 1)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Volume'], label='Original Volume', color='blue')
plt.scatter(sp500_data_filtered.index[sp500_data_filtered['Volume_outlier']],
            sp500_data_filtered['Volume'][sp500_data_filtered['Volume_outlier']],
            color='red', label='Detected Outliers')
plt.title('Volume Before Correction')
plt.legend()

# 子图2: 修复后的交易量
plt.subplot(2, 1, 2)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Volume_corrected'], label='Corrected Volume', color='green')
plt.title('Volume After Correction')
plt.legend()

plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'volume_before_after_correction.svg')
plt.clf()  # 清除当前图像

#%% 7. 短期与长期趋势

# 时间窗口划分
short_window = 20  # 短期窗口（如20天）
long_window = 200  # 长期窗口（如200天）

# 计算短期和长期的移动平均
sp500_data_filtered['Short_MA'] = sp500_data_filtered['Close'].rolling(window=short_window, min_periods=1).mean()
sp500_data_filtered['Long_MA'] = sp500_data_filtered['Close'].rolling(window=long_window, min_periods=1).mean()

# 计算短期和长期的波动性（使用移动标准差）
sp500_data_filtered['Short_Volatility'] = sp500_data_filtered['Close'].rolling(window=short_window, min_periods=1).std()
sp500_data_filtered['Long_Volatility'] = sp500_data_filtered['Close'].rolling(window=long_window, min_periods=1).std()

# 可视化短期与长期的移动平均与波动性
plt.figure(figsize=(10, 6))

# 子图1: 收盘价与短期、长期移动平均线
plt.subplot(2, 1, 1)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Close'], label='Close Price', color='blue')
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Short_MA'], label=f'{short_window}-Day MA', color='orange')
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Long_MA'], label=f'{long_window}-Day MA', color='green')
plt.title(f'S&P 500 Close Price with {short_window}-Day and {long_window}-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 子图2: 短期与长期波动性
plt.subplot(2, 1, 2)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Short_Volatility'], label=f'{short_window}-Day Volatility', color='orange')
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Long_Volatility'], label=f'{long_window}-Day Volatility', color='green')
plt.title(f'S&P 500 {short_window}-Day and {long_window}-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()

plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'short_long_trend_volatility.svg')
plt.clf()  # 清除当前图像

#%% 8. 平滑处理

# 计算技术指标
sp500_data_filtered['SMA_20'] = sp500_data_filtered['Close'].rolling(window=20).mean()  # 20天简单移动平均线
sp500_data_filtered['EMA_20'] = sp500_data_filtered['Close'].ewm(span=20, adjust=False).mean()  # 20天指数移动平均线

# 可视化技术指标
plt.figure(figsize=(10, 6))
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Close'], label='Close Price', color='blue')
plt.plot(sp500_data_filtered.index, sp500_data_filtered['SMA_20'], label='20-day SMA', color='green')
plt.plot(sp500_data_filtered.index, sp500_data_filtered['EMA_20'], label='20-day EMA', color='orange')
plt.title('S&P 500 Close Price with Technical Indicators (SMA, EMA)')
plt.legend()
plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'close_price_with_indicators.svg')
plt.clf()  # 清除当前图像

#%% 9. 滞后特征

# 计算滞后特征，分别为滞后1天、5天、10天的收盘价
sp500_data_filtered['Lag_1'] = sp500_data_filtered['Close'].shift(1)
sp500_data_filtered['Lag_5'] = sp500_data_filtered['Close'].shift(5)
sp500_data_filtered['Lag_10'] = sp500_data_filtered['Close'].shift(10)

# 线性插值填充 NaN
sp500_data_filtered['Lag_1'].interpolate(method='linear', inplace=True)
sp500_data_filtered['Lag_5'].interpolate(method='linear', inplace=True)
sp500_data_filtered['Lag_10'].interpolate(method='linear', inplace=True)

# 可视化当前收盘价与滞后特征
plt.figure(figsize=(8, 6))

# 设置时间范围，选择数据较为集中的区间进行可视化
start_date = '2010-01-01'
end_date = '2010-01-31'

# 过滤时间范围内的数据
sp500_data_zoomed = sp500_data_filtered.loc[start_date:end_date]

# 子图1: 当前收盘价与滞后1天的收盘价
plt.subplot(3, 1, 1)
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Close'], label='Close Price', color='blue', marker='o')
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Lag_1'], label='Lag 1 Day', color='orange', marker='o')
plt.title('S&P 500 Close Price vs. Lag 1 Day (2010-01)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 子图2: 当前收盘价与滞后5天的收盘价
plt.subplot(3, 1, 2)
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Close'], label='Close Price', color='blue', marker='o')
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Lag_5'], label='Lag 5 Days', color='green', marker='o')
plt.title('S&P 500 Close Price vs. Lag 5 Days (2010-01)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 子图3: 当前收盘价与滞后10天的收盘价
plt.subplot(3, 1, 3)
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Close'], label='Close Price', color='blue', marker='o')
plt.plot(sp500_data_zoomed.index, sp500_data_zoomed['Lag_10'], label='Lag 10 Days', color='red', marker='o')
plt.title('S&P 500 Close Price vs. Lag 10 Days (2010-01)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 整理布局并保存图片
plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'lag_features_visualization.svg')
plt.clf()  # 清除当前图像

#%% 10. 交叉特征

# 1. 计算价格波动率与交易量的乘积
sp500_data_filtered['Volatility_Volume_Product'] = sp500_data_filtered['Volatility'] * sp500_data_filtered['Volume']

# 2. 计算利率变化率与市场回报率的比值
sp500_data_filtered['Interest_Rate_Change'] = sp500_data_filtered['Interest_Rate'].diff()
# 处理可能的除零情况
sp500_data_filtered['Interest_Rate_to_Return'] = np.where(
    sp500_data_filtered['Log_Return'] != 0,
    sp500_data_filtered['Interest_Rate_Change'] / sp500_data_filtered['Log_Return'],
    0  # 当Log_Return为0时，将结果设为0
)

# 3. 可视化交叉特征
plt.figure(figsize=(8, 6))

# 子图1: 价格波动率与交易量的乘积
plt.subplot(2, 1, 1)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Volatility_Volume_Product'], label='Volatility * Volume',
         color='purple')
plt.title('Volatility and Volume Product Over Time')
plt.xlabel('Date')
plt.ylabel('Volatility * Volume')
plt.legend()

# 子图2: 利率变化率与市场回报率的比值
plt.subplot(2, 1, 2)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Interest_Rate_to_Return'],
         label='Interest Rate Change / Market Return', color='brown')
plt.title('Interest Rate Change to Market Return Ratio Over Time')
plt.xlabel('Date')
plt.ylabel('Interest Rate Change / Market Return')
plt.legend()

plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'cross_features_visualization.svg')
plt.clf()  # 清除当前图像

#%% 11. 归一化

# 选定要归一化的特征
features_to_normalize = ['Close', 'Volume', 'Interest_Rate', 'Inflation_Rate']

# 创建 Min-Max Scaler
scaler = MinMaxScaler()

# 对选定的特征进行归一化
df_normalized = sp500_data_filtered.copy()  # 保留原数据集
df_normalized[features_to_normalize] = scaler.fit_transform(sp500_data_filtered[features_to_normalize])

# 可视化归一化后的收盘价与交易量
plt.figure(figsize=(8, 6))

# 归一化后的收盘价曲线
plt.subplot(2, 1, 1)
plt.plot(df_normalized.index, df_normalized['Close'], label='Normalized Close Price', color='blue')
plt.title('Normalized S&P 500 Close Price')
plt.xlabel('Date')
plt.ylabel('Normalized Close Price')
plt.legend()

# 归一化后的交易量曲线
plt.subplot(2, 1, 2)
plt.plot(df_normalized.index, df_normalized['Volume'], label='Normalized Volume', color='green')
plt.title('Normalized S&P 500 Trading Volume')
plt.xlabel('Date')
plt.ylabel('Normalized Volume')
plt.legend()

plt.tight_layout()
plt.savefig(EXPLORATION_DIR / 'normalized_features_plot.svg')
plt.clf()  # 清除当前图像

#%% 12. 特征保存

# 清理和重置数据索引以进行保存
sp500_data_filtered = sp500_data_filtered.reset_index()
# 填充缺失值
sp500_data_filtered.fillna(0, inplace=True)
# 删除无限值
sp500_data_filtered.replace([np.inf, -np.inf], 0, inplace=True)

# 保存处理后的特征数据集
sp500_data_filtered.to_csv(PROCESSED_DATA_DIR / 'sp500_features.csv', index=False)

print(f"特征工程已完成，已保存处理后的数据到：{PROCESSED_DATA_DIR / 'sp500_features.csv'}")
print(f"生成的特征总数：{len(sp500_data_filtered.columns) - 1}")  # 减去日期列
print("特征列表:")
for col in sp500_data_filtered.columns:
    if col != 'Date':
        print(f"- {col}")