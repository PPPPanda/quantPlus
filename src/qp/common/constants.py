"""
项目级常量定义.

集中管理所有映射表和常量，避免分散在各模块中重复定义。
"""

from __future__ import annotations

from vnpy.trader.constant import Exchange, Interval

# 品种代码映射 (vnpy symbol -> akshare symbol)
SYMBOL_MAP: dict[str, str] = {
    "p0": "P0",      # 棕榈油连续
    "p": "P0",       # 棕榈油
    "y0": "Y0",      # 豆油连续
    "m0": "M0",      # 豆粕连续
    "c0": "C0",      # 玉米连续
    "i0": "I0",      # 铁矿石连续
    "rb0": "RB0",    # 螺纹钢连续
    "hc0": "HC0",    # 热卷连续
    "cu0": "CU0",    # 铜连续
    "al0": "AL0",    # 铝连续
    "zn0": "ZN0",    # 锌连续
    "ag0": "AG0",    # 白银连续
    "au0": "AU0",    # 黄金连续
}

# 交易所映射 (字符串 -> Exchange 枚举)
EXCHANGE_MAP: dict[str, Exchange] = {
    "DCE": Exchange.DCE,      # 大连商品交易所
    "SHFE": Exchange.SHFE,    # 上海期货交易所
    "CZCE": Exchange.CZCE,    # 郑州商品交易所
    "CFFEX": Exchange.CFFEX,  # 中国金融期货交易所
    "INE": Exchange.INE,      # 上海国际能源交易中心
    "SSE": Exchange.SSE,      # 上海证券交易所
    "SZSE": Exchange.SZSE,    # 深圳证券交易所
    "BSE": Exchange.BSE,      # 北京证券交易所
}

# 周期映射 (字符串 -> Interval 枚举)
INTERVAL_MAP: dict[str, Interval] = {
    # 全称
    "MINUTE": Interval.MINUTE,
    "HOUR": Interval.HOUR,
    "DAILY": Interval.DAILY,
    "WEEKLY": Interval.WEEKLY,
    # 缩写
    "1m": Interval.MINUTE,
    "1h": Interval.HOUR,
    "1d": Interval.DAILY,
    "1w": Interval.WEEKLY,
    # GUI 格式
    "d": Interval.DAILY,
    "h": Interval.HOUR,
    "m": Interval.MINUTE,
    "w": Interval.WEEKLY,
}

# 中国期货交易所列表（用于判断是否需要使用 akshare）
CHINA_FUTURES_EXCHANGES: set[str] = {"DCE", "SHFE", "CZCE", "CFFEX", "INE"}

# 策略 ArrayManager 默认缓冲区大小
AM_BUFFER_SIZE: int = 30
