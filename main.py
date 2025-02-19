import yfinance as yf
from datetime import datetime
from src import RAW_DATA_DIR, RESULTS_DIR
from src.data_analysis import load_data, calculate_basic_features, analyze_features, plot_analysis


def fetch_stock_data(symbol="^GSPC", start_date="2019-01-01"):
    """
    获取股票数据并保存
    """
    try:
        print(f"正在下载 {symbol} 的数据...")
        df = yf.download(symbol, start=start_date)

        # 保存原始数据
        file_path = RAW_DATA_DIR / f"{symbol.replace('^', '')}_data.csv"
        df.to_csv(file_path)
        print(f"数据已保存到: {file_path}")

        return True

    except Exception as e:
        print(f"下载数据时发生错误: {str(e)}")
        return False


def main():
    """
    主程序入口
    """
    print("开始数据分析...")

    # 1. 获取数据
    if fetch_stock_data():
        # 2. 加载数据
        df = load_data()

        # 3. 计算特征
        df = calculate_basic_features(df)

        # 4. 分析特征
        stats = analyze_features(df)
        print("\n基本统计信息:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 5. 绘制分析图表
        plot_path = RESULTS_DIR / 'analysis_plots.png'
        plot_analysis(df, save_path=plot_path)

        print(f"\n分析图表已保存到: {plot_path}")

    print("\n分析完成!")


if __name__ == "__main__":
    main()