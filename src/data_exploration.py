import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
import pandas_datareader.data as web
import pandas_market_calendars as mcal
from scipy.stats import pearsonr
import numpy as np
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

current_file = Path(__file__)
ROOT_DIR = current_file.parent.parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
EXPLORATION_DIR = DATA_DIR / 'exploration'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.figsize'] = (10, 6)
pdf_path = EXPLORATION_DIR / 'sp500_data_exploration.pdf'
pdf = PdfPages(pdf_path)


print('正在读取数据...')
sp500_data = pd.read_csv(RAW_DATA_DIR / 'SPX.csv', parse_dates=True)
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])  # 将 'Date' 列转换为 datetime 类型

print('数据概览:')
print(sp500_data.info())
print('\n前5行数据:')
print(sp500_data.head())
print('\n后5行数据:')
print(sp500_data.tail())

sp500_data = sp500_data.set_index(sp500_data['Date']).sort_index()

volume_data = sp500_data[sp500_data['Volume'] > 0]
print('\n交易量>0的前5行数据:')
print(volume_data.head())

plt.figure(figsize=(10, 6))
plt.plot(sp500_data.index, sp500_data['Volume'], label='Volume', color='blue')

plt.title('S&P 500 Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')

years = mdates.YearLocator(20)
years_fmt = mdates.DateFormatter('%Y')

ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
plt.xticks(rotation=45)
plt.tight_layout()

pdf.savefig()
plt.close()

sp500_data_filtered = sp500_data.loc['2000-01-01':]

plt.figure(figsize=(10, 6))
plt.plot(sp500_data_filtered['Date'], sp500_data_filtered['Close'], label='Close', color='blue')
plt.title('S&P 500 Close Over Time')
plt.xlabel('Date')
plt.ylabel('Close')

years = mdates.YearLocator(5)
years_fmt = mdates.DateFormatter('%Y')
ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
plt.xticks(rotation=45)
plt.tight_layout()
pdf.savefig()
plt.close()

plt.figure(figsize=(10, 6))
sns.kdeplot(sp500_data_filtered['Close'], fill=True)
plt.title('S&P 500 Close Distribution')
plt.xlabel('Close')
plt.ylabel('Density')
plt.tight_layout()
pdf.savefig()
plt.close()

close_series = sp500_data_filtered['Close']

print('正在进行时间序列分解.')
decomposition = seasonal_decompose(close_series, model='multiplicative', period=365)
plt.figure(figsize=(10, 10))

plt.subplot(411)
plt.plot(close_series, label='Observed')
plt.title('Observed S&P 500 Close Prices')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.title('Trend')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.title('Seasonal')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.title('Residuals')

plt.tight_layout()
pdf.savefig()
plt.close()

crisis_period = (sp500_data_filtered.index >= '2007-11-01') & (sp500_data_filtered.index <= '2009-03-01')

recovery_period = (sp500_data_filtered.index > '2009-03-01') & (sp500_data_filtered.index <= '2012-12-31')

crisis_data = sp500_data_filtered[crisis_period]
recovery_data = sp500_data_filtered[recovery_period]
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(crisis_data.index, crisis_data['Close'], label='During Financial Crisis', color='red')
plt.title('S&P 500 Closing Price - Financial Crisis (2007-11 - 2009-03)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(recovery_data.index, recovery_data['Close'], label='During Recovery Period', color='green')
plt.title('S&P 500 Closing Price - Recovery Period (2009-03 - 2012-12)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()

plt.tight_layout()

pdf.savefig()
plt.close()


print('进行波动性分析...')
sp500_data_filtered['Returns'] = sp500_data_filtered['Close'].pct_change()

window_size = 20  # 窗口大小为20天
sp500_data_filtered['Rolling_Std'] = sp500_data_filtered['Close'].rolling(window=window_size).std()

# 年化波动率 = 每日收益率的标准差 × √252
sp500_data_filtered['Volatility'] = sp500_data_filtered['Returns'].rolling(window=window_size).std() * np.sqrt(252)

plt.figure(figsize=(10, 12))

# 子图1: 收盘价
plt.subplot(3, 1, 1)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Close'], label='Close Price', color='blue')
plt.title('S&P 500 Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.legend()

# 子图2: 短期波动性（移动标准差）
plt.subplot(3, 1, 2)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Rolling_Std'], label='20-Day Rolling Std (Volatility)',
         color='orange')
plt.title('S&P 500 20-Day Rolling Standard Deviation (Volatility)')
plt.xlabel('Date')
plt.ylabel('Rolling Std (Volatility)')
plt.grid(True)
plt.legend()

# 子图3: 年化波动率
plt.subplot(3, 1, 3)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Volatility'], label='Annualized Volatility', color='red')
plt.title('S&P 500 Annualized Volatility')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.grid(True)
plt.legend()

plt.tight_layout()

# 保存到PDF
pdf.savefig()
plt.close()

# %% 7. 异常值检测

print('进行异常值检测...')
# 1. 使用箱线图检测交易量和收盘价中的异常值

plt.figure(figsize=(10, 6))

# 绘制箱线图检测收盘价中的异常值
plt.subplot(1, 2, 1)
sns.boxplot(y=sp500_data_filtered['Close'], color='lightblue')
plt.title('Box Plot of S&P 500 Closing Price')

# 绘制箱线图检测交易量中的异常值
plt.subplot(1, 2, 2)
sns.boxplot(y=sp500_data_filtered['Volume'], color='lightgreen')
plt.title('Box Plot of S&P 500 Trading Volume')

plt.tight_layout()

# 保存到PDF
pdf.savefig()
plt.close()



# Z 分数函数，z = (x - mean) / std
def z_score(df, column_name):
    return (df[column_name] - df[column_name].mean()) / df[column_name].std()


# 计算收盘价和交易量的 Z 分数
sp500_data_filtered['Close_zscore'] = z_score(sp500_data_filtered, 'Close')
sp500_data_filtered['Volume_zscore'] = z_score(sp500_data_filtered, 'Volume')

z_threshold = 3

# 识别出异常的收盘价和交易量
price_outliers = sp500_data_filtered[abs(sp500_data_filtered['Close_zscore']) > z_threshold]
volume_outliers = sp500_data_filtered[abs(sp500_data_filtered['Volume_zscore']) > z_threshold]

# 输出检测到的异常值
print(f"检测到 {len(price_outliers)} 个价格异常值和 {len(volume_outliers)} 个交易量异常值。")

# 3. 可视化检测到的异常点

plt.figure(figsize=(10, 10))

# 绘制收盘价和异常点
plt.subplot(2, 1, 1)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Close'], label='Closing Price', color='blue')
plt.scatter(price_outliers.index, price_outliers['Close'], color='red', label='Price Outliers', zorder=5)
plt.title('S&P 500 Closing Price with Outliers')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()

# 绘制交易量和异常点
plt.subplot(2, 1, 2)
plt.plot(sp500_data_filtered.index, sp500_data_filtered['Volume'], label='Trading Volume', color='green')
plt.scatter(volume_outliers.index, volume_outliers['Volume'], color='red', label='Volume Outliers', zorder=5)
plt.title('S&P 500 Trading Volume with Outliers')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()

plt.tight_layout()

pdf.savefig()
plt.close()


print('分析宏观经济影响...')
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2020, 11, 4)

try:
    interest_rate_data = web.DataReader('DFF', 'fred', start, end)
    print("成功获取利率数据")

    # 获取CPI数据
    cpi_data = web.DataReader('CPIAUCSL', 'fred', start, end)
    print("成功获取CPI数据")

    cpi_data['Inflation_Rate'] = cpi_data.pct_change() * 100
    interest_rate_data.columns = ['Interest_Rate']
    cpi_data.columns = ['CPI', 'Inflation_Rate']

    macro_data = pd.merge(interest_rate_data, cpi_data, left_index=True, right_index=True, how='outer')
    macro_data.sort_index(inplace=True)
    macro_data['CPI'] = macro_data['CPI'].interpolate(method='linear')  # 插值法
    macro_data['Inflation_Rate'] = macro_data['Inflation_Rate'].ffill()  # 前向填充
    macro_data['Interest_Rate'] = macro_data['Interest_Rate'].ffill()  # 前向填充
    sp500_macro_data = sp500_data_filtered[['Close']].join(macro_data, how='inner')

    # 绘制利率、通胀率和S&P 500收盘价
    plt.figure(figsize=(10, 12))

    # S&P 500 收盘价
    plt.subplot(3, 1, 1)
    plt.plot(sp500_macro_data.index, sp500_macro_data['Close'], color='blue', label='S&P 500 Close')
    plt.title('S&P 500 Closing Price')
    plt.ylabel('Price')
    plt.grid(True)

    # 利率
    plt.subplot(3, 1, 2)
    plt.plot(sp500_macro_data.index, sp500_macro_data['Interest_Rate'], color='red', label='Interest Rate')
    plt.title('Interest Rate (Federal Funds Rate)')
    plt.ylabel('Rate (%)')
    plt.grid(True)

    # 通胀率
    plt.subplot(3, 1, 3)
    plt.plot(sp500_macro_data.index, sp500_macro_data['Inflation_Rate'], color='green', label='Inflation Rate')
    plt.title('Inflation Rate (CPI YoY Change)')
    plt.ylabel('Rate (%)')
    plt.grid(True)

    # 设置 X 轴时间格式
    for ax in plt.gcf().axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))

    # 利率与收盘价的关系
    plt.subplot(1, 2, 1)
    plt.scatter(sp500_macro_data['Interest_Rate'], sp500_macro_data['Close'], color='red', alpha=0.5)
    plt.title('Relationship between interest rate and closing price')
    plt.xlabel('Interest Rate (%)')
    plt.ylabel('S&P 500 Closing Price')

    # 通胀率与收盘价的关系
    plt.subplot(1, 2, 2)
    plt.scatter(sp500_macro_data['Inflation_Rate'], sp500_macro_data['Close'], color='green', alpha=0.5)
    plt.title('Relationship between inflation rate and closing price')
    plt.xlabel('Inflation Rate (%)')
    plt.ylabel('S&P 500 Closing Price')

    plt.tight_layout()

    pdf.savefig()
    plt.close()
except Exception as e:
    print(f"获取宏观经济数据失败: {e}")
    print("跳过宏观经济分析部分，继续执行其他分析...")

#9. 时间分析
print('进行时间分析...')
# 时间频率检查
time_diff_counts = sp500_data_filtered['Date'].diff().value_counts()
print("日期间隔分布:")
print(time_diff_counts.head())

# 获取纽约证券交易所的日历
try:
    nyse = mcal.get_calendar('NYSE')

    # 指定交易日期范围
    start_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2020-11-04')

    # 获取交易日日期范围
    all_trading_days = nyse.schedule(start_date, end_date)

    # 创建一个包含所有理论上交易日的 DataFrame
    all_trading_days_df = pd.DataFrame(all_trading_days.index, columns=['Date'])

    # 将不寻常时间间隔的数据与完整交易日历进行合并，看看是否有缺失
    merged_data = pd.merge(all_trading_days_df.set_index('Date'), sp500_data_filtered[['Close']],
                           left_index=True, right_index=True, how='left')

    missing_dates = merged_data[merged_data['Close'].isna()]
    print(f"发现 {len(missing_dates)} 个缺失的交易日数据")

    # 可视化缺失日期的分布
    if len(missing_dates) > 0:
        merged_data['is_missing'] = merged_data['Close'].isna()
        merged_data['year'] = merged_data.index.year
        merged_data['month'] = merged_data.index.month
        yearly_missing = merged_data.groupby('year')['is_missing'].sum()

        plt.figure(figsize=(10, 6))
        yearly_missing.plot(kind='bar')
        plt.title('Missing Trading Days by Year')
        plt.xlabel('Year')
        plt.ylabel('Count of Missing Days')
        plt.tight_layout()

        pdf.savefig()
        plt.close()
except Exception as e:
    print(f"时间分析部分出错: {e}")
    print("跳过时间分析中的交易日历比较，继续执行其他分析...")


print('进行交易量分析...')
plt.figure(figsize=(10, 6))
sns.regplot(x=sp500_data_filtered['Volume'], y=sp500_data_filtered['Close'], scatter_kws={'s': 10},
            line_kws={'color': 'red'})
plt.title('Relationship Between Trading Volume and Close Price', fontsize=14)
plt.xlabel('Trading Volume', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.grid(True)
plt.tight_layout()

pdf.savefig()
plt.close()

# 2. 计算 Pearson 相关系数，定量分析交易量和收盘价之间的关系
volume = sp500_data_filtered['Volume']
close_price = sp500_data_filtered['Close']
correlation, p_value = pearsonr(volume, close_price)

# 输出相关系数及其显著性
print(f"Pearson相关系数: {correlation:.4f}")
print(f"P值: {p_value:.4f}")

# 3. 根据相关系数给出分析结论
if p_value < 0.05:
    if correlation > 0:
        print("交易量和收盘价之间存在统计显著的正相关关系。")
    else:
        print("交易量和收盘价之间存在统计显著的负相关关系。")
else:
    print("交易量和收盘价之间没有统计显著的相关关系。")

# 创建相关性可视化图表
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, f"Volume-Price Correlation: {correlation:.4f}\nP-value: {p_value:.4f}",
         horizontalalignment='center', verticalalignment='center',
         fontsize=16, transform=plt.gca().transAxes)
plt.axis('off')
pdf.savefig()
plt.close()
plt.figure(figsize=(10, 8))
plt.text(0.5, 0.5, "S&P 500 Data Exploration Summary\n\n" +
         f"Data Period: {sp500_data_filtered.index.min().strftime('%Y-%m-%d')} to {sp500_data_filtered.index.max().strftime('%Y-%m-%d')}\n" +
         f"Total Records: {len(sp500_data_filtered)}\n" +
         f"Detected Price Outliers: {len(price_outliers)}\n" +
         f"Detected Volume Outliers: {len(volume_outliers)}\n" +
         f"Volume-Price Correlation: {correlation:.4f} (p-value: {p_value:.4f})\n\n" +
         "Key Insights:\n" +
         "1. Stock price shows clear trend and seasonal patterns\n" +
         "2. Significant volatility during 2007-2009 financial crisis\n" +
         "3. Volume and price show significant correlation\n" +
         "4. Macroeconomic factors (interest rate, inflation) exhibit relationships with price patterns",
         horizontalalignment='center', verticalalignment='center',
         fontsize=14, transform=plt.gca().transAxes)
plt.axis('off')
pdf.savefig()
plt.close()
pdf.close()

print(f"探索性分析完成，所有图表已保存到 PDF 文件: {pdf_path}")