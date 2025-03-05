# Stock Movement Prediction with LSTM

A comparative study on single vs stacked LSTM architectures for short-term stock movement prediction, focusing on S&P 500 stocks with comprehensive data exploration and feature engineering.

## Project Overview

This project explores the effectiveness of different LSTM model architectures for predicting stock price movements. We conduct thorough data exploration, implement detailed feature engineering, and compare single-layer vs double-layer LSTM models with various configurations.

## Project Structure
```
ML_LSTM/
├── data/
│   ├── raw/          # SPX.csv and other raw financial data
│   ├── exploration/  # Data exploration visualizations and results
│   └── processed/    # Processed feature datasets
│
├── models/           # Saved trained models
│
├── results/          # Model prediction results and performance metrics
│
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── data_exploration.py          # Exploratory data analysis
│   ├── data_feature_engineering.py  # Feature extraction and processing
│   ├── model_building.py            # Base LSTM model implementation
│   ├── model_building_single_layer.py # Single-layer LSTM implementation
│   ├── model_improvement.py         # Enhanced LSTM with regularization
│   ├── model_timestep_analysis.py   # Analysis of time window effects
│   └── model_complexity_analysis.py # Analysis of model architecture effects
│
├── main.py                # Main execution script 
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Features
- Comprehensive exploratory data analysis of S&P 500 historical data
- Advanced feature engineering including:
  - Time series decomposition (trend, seasonality, residual)
  - Volatility features (rolling standard deviation, annualized volatility)
  - Trading volume analysis and outlier detection/correction
  - Macroeconomic factor integration (interest rates, inflation)
  - Technical indicators (moving averages, EMA)
  - Lag features and cross-features
- LSTM model implementations with various configurations:
  - Single-layer vs double-layer architecture comparison
  - Different neuron counts (50, 200)
  - Regularization techniques (Dropout, L2)
  - Time window optimization (10-120 days)

## Key Findings
- Clear cyclical patterns and trend components in S&P 500 data
- Significant volatility during crisis periods (2007-2009, 2020)
- Statistical correlation between trading volume and price (0.24)
- Relationship between macroeconomic factors and stock prices
- Time window of 60 days provides optimal prediction performance

## Dependencies
Main dependencies include:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- statsmodels
- scikit-learn
- pandas_datareader
- pandas_market_calendars

## Current Progress
- [x] Project setup and structure
- [x] Data collection and preprocessing
- [x] Comprehensive data exploration and visualization
- [x] Advanced feature engineering
- [x] Implementation of baseline LSTM models (single & double layer)
- [x] Model improvement with regularization techniques
- [x] Time window size analysis
- [x] Model complexity analysis
- [ ] Hyperparameter optimization
- [ ] Final model evaluation and comparison
- [ ] Performance analysis in different market conditions

## Future Work
- Ensemble methods combining multiple LSTM configurations
- Integration of sentiment analysis from financial news
- Attention mechanisms for improved feature importance
- Extension to other market indices and individual stocks