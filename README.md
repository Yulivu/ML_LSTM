# ML_175
# Stock Movement Prediction with LSTM

A comparative study on single vs stacked LSTM architectures for short-term stock movement prediction, focusing on S&P 500 stocks.

## Project Structure
```
ML_LSTM/
├── data/
│   ├── raw/          # SPX.csv
│   ├── exploration/  # 存放EDA结果
│   └── processed/    # 存放处理后的特征数据
│
├── models/           # 保存训练好的模型
│
├── results/          # 模型预测结果和性能评估
│
├── src/
│   ├── __init__.py   # 包初始化
│   ├── data_exploration.py        # 原1_EDA.py
│   ├── data_feature_engineering.py # 原2_feature_engineering.py
│   ├── model_building.py           # 原1_build_model.py
│   ├── model_improvement.py        # 原2_improve_model.py
│   ├── model_timestep_analysis.py  # 原3_time_step_model_performance.py
│   └── model_complexity_analysis.py # 原4_model_complexity.py
├── main.py       # 主执行脚本 
├── requirements.txt  # 项目依赖
└── README.md         # 项目文档
```

## Features
- Stock data retrieval using yfinance API
- Basic data analysis and visualization
- Technical indicator calculations
- LSTM model implementation (upcoming)

## Dependencies
Main dependencies include:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- yfinance
- scikit-learn

## Objective
Compare the effectiveness of single LSTM vs stacked LSTM architectures in predicting stock market movements, focusing on directional (up/down) prediction rather than precise price forecasting.

## Current Progress
- [x] Project setup and structure
- [x] Data collection pipeline
- [x] Basic data analysis and visualization
- [x] Feature engineering
- [x] LSTM model implementation
- [x] Model comparison and evaluation