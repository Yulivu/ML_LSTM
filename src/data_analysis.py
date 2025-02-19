import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_data(symbol="GSPC"):
    """
    加载原始数据
    """
    file_path = RAW_DATA_DIR / f"{symbol}_data.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def calculate_basic_features(df):
    """
    计算基本特征
    """
    # 计算日收益率
    df['Daily_Return'] = df['Close'].pct_change()

    # 计算移动平均线
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # 计算波动率（20日）
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    return df


def analyze_features(df):
    """
    分析特征，返回关键统计信息
    """
    stats = {
        'date_range': f"{df.index[0]} to {df.index[-1]}",
        'trading_days': len(df),
        'avg_volume': df['Volume'].mean(),
        'avg_daily_return': df['Daily_Return'].mean(),
        'volatility': df['Daily_Return'].std()
    }
    return stats


def plot_analysis(df, save_path=None):
    """
    绘制分析图表
    """
    plt.style.use('seaborn')

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 价格和移动平均线
    axes[0, 0].plot(df.index, df['Close'], label='Price')
    axes[0, 0].plot(df.index, df['MA20'], label='20-day MA')
    axes[0, 0].plot(df.index, df['MA50'], label='50-day MA')
    axes[0, 0].set_title('Price and Moving Averages')
    axes[0, 0].legend()

    # 2. 成交量
    axes[0, 1].bar(df.index, df['Volume'], alpha=0.5)
    axes[0, 1].set_title('Trading Volume')

    # 3. 收益率分布
    sns.histplot(df['Daily_Return'].dropna(), bins=50, ax=axes[1, 0])
    axes[1, 0].set_title('Daily Returns Distribution')

    # 4. 波动率
    axes[1, 1].plot(df.index, df['Volatility'])
    axes[1, 1].set_title('20-day Rolling Volatility')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()