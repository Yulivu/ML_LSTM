# ML_175
# Stock Movement Prediction with LSTM

A comparative study on single vs stacked LSTM architectures for short-term stock movement prediction, focusing on S&P 500 stocks.

## Project Structure
```
ML_LSTM/
├── data/
│   ├── raw/          # Original stock data
│   └── processed/    # Processed and prepared datasets
├── models/           # Saved LSTM models
├── results/          # Analysis outputs and visualizations
├── src/
│   ├── __init__.py   # Package initialization and path configurations
│   ├── data_analysis.py  # Data analysis and visualization functions
│   └── main.py       # Main script for running the analysis
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
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
- [ ] Data collection pipeline
- [ ] Basic data analysis and visualization
- [ ] Feature engineering
- [ ] LSTM model implementation
- [ ] Model comparison and evaluation