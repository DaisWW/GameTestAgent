"""pytest 配置：将项目根目录加入 sys.path，使所有模块可直接 import。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
