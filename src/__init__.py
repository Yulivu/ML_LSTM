from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

# 确保所有必要的目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# define time & data
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_SYMBOL = "^GSPC"  # S&P 500

# define path
def get_raw_data_path(symbol):
    """Get raw data path"""
    return RAW_DATA_DIR / f"{symbol.replace('^', '')}_data.csv"

def get_processed_data_path(symbol):
    """Get processed data path"""
    return PROCESSED_DATA_DIR / f"{symbol.replace('^', '')}_processed.csv"