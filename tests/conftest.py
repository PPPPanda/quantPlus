"""
pytest 配置和共享 fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 Python 路径中
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)


@pytest.fixture
def sample_vt_symbols() -> list[str]:
    """返回测试用的 vt_symbol 列表."""
    return [
        "p0.DCE",
        "p2501.DCE",
        "rb0.SHFE",
        "IF2501.CFFEX",
        "cu0.SHFE",
    ]


@pytest.fixture
def invalid_vt_symbols() -> list[str]:
    """返回无效的 vt_symbol 列表."""
    return [
        "p0",           # 缺少交易所
        ".DCE",         # 缺少 symbol
        "p0.INVALID",   # 无效交易所
        "",             # 空字符串
    ]
