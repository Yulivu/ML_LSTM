# 设置项目根目录和重要路径
import os
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 添加项目根目录到Python路径
sys.path.append(str(ROOT_DIR))

# 定义常用目录
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
EXPLORATION_DIR = DATA_DIR / 'exploration'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

# 确保所有需要的目录都存在
for directory in [RAW_DATA_DIR, EXPLORATION_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)