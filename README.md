# Stock Movement Prediction with LSTM

A comparative study on single vs stacked LSTM architectures for short-term stock movement prediction, focusing on S&P 500 stocks with comprehensive data exploration and feature engineering.

## Project Overview

This project explores the effectiveness of different LSTM model architectures for predicting stock price movements. We conduct thorough data exploration, implement detailed feature engineering, and compare single-layer vs double-layer LSTM models with various configurations.

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

## Future Work
- Ensemble methods combining multiple LSTM configurations
- Integration of sentiment analysis from financial news
- Attention mechanisms for improved feature importance
- Extension to other market indices and individual stocks