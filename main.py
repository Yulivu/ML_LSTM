"""
LSTM股价预测模型主执行脚本
"""
import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction')
    parser.add_argument('--step', type=str, default='all',
                        choices=['exploration', 'feature', 'build', 'improve', 'timestep', 'complexity', 'all'],
                        help='Specify which step to run')

    args = parser.parse_args()

    if args.step == 'exploration' or args.step == 'all':
        print("执行数据探索分析...")
        from src import data_exploration

    if args.step == 'feature' or args.step == 'all':
        print("执行特征工程...")
        from src import data_feature_engineering

    if args.step == 'build' or args.step == 'all':
        print("构建基础模型...")
        from src import model_building

    if args.step == 'improve' or args.step == 'all':
        print("改进模型...")
        from src import model_improvement

    if args.step == 'timestep' or args.step == 'all':
        print("分析时间窗口大小影响...")
        from src import model_timestep_analysis

    if args.step == 'complexity' or args.step == 'all':
        print("分析模型复杂度影响...")
        from src import model_complexity_analysis

    print("执行完成！")


if __name__ == "__main__":
    main()