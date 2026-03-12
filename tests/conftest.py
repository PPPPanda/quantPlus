"""
pytest conftest: 在 WSL/Linux 环境下 mock 掉 vnpy 依赖.

vnpy 只安装在 Windows .venv-win 中，Linux 测试环境需要 mock。
"""

import sys
from unittest.mock import MagicMock


def _ensure_vnpy_mock():
    """如果 vnpy 不可用，注入 mock 到 sys.modules."""
    try:
        import vnpy  # noqa: F401
    except ImportError:
        # Mock vnpy 及其子模块
        vnpy_mock = MagicMock()
        modules = [
            "vnpy",
            "vnpy.event",
            "vnpy.trader",
            "vnpy.trader.engine",
            "vnpy.trader.constant",
            "vnpy.trader.object",
            "vnpy.trader.event",
            "vnpy.trader.utility",
            "vnpy.trader.setting",
            "vnpy.trader.database",
            "vnpy_ctp",
            "vnpy_tts",
            "vnpy_ctastrategy",
            "vnpy_ctastrategy.base",
            "vnpy_datarecorder",
            "vnpy_datarecorder.engine",
            "vnpy_spreadtrading",
            "vnpy_spreadtrading.base",
            "vnpy_sqlite",
            "vnpy_sqlite.sqlite_database",
        ]
        for mod in modules:
            sys.modules[mod] = vnpy_mock

        # 设置 Exchange 和 Interval 枚举的 mock
        from enum import Enum

        class MockExchange(str, Enum):
            CFFEX = "CFFEX"
            SHFE = "SHFE"
            DCE = "DCE"
            CZCE = "CZCE"
            INE = "INE"
            SSE = "SSE"
            SZSE = "SZSE"
            BSE = "BSE"
            GFEX = "GFEX"
            LOCAL = "LOCAL"

        class MockInterval(str, Enum):
            MINUTE = "1m"
            HOUR = "1h"
            DAILY = "d"
            WEEKLY = "w"

        class MockDirection(str, Enum):
            LONG = "多"
            SHORT = "空"

        class MockOffset(str, Enum):
            NONE = ""
            OPEN = "开"
            CLOSE = "平"
            CLOSETODAY = "平今"
            CLOSEYESTERDAY = "平昨"

        class MockStatus(str, Enum):
            SUBMITTING = "提交中"
            NOTTRADED = "未成交"
            PARTTRADED = "部分成交"
            ALLTRADED = "全部成交"
            CANCELLED = "已撤销"
            REJECTED = "拒单"

        class MockOrderType(str, Enum):
            LIMIT = "限价"
            MARKET = "市价"
            STOP = "STOP"
            FAK = "FAK"
            FOK = "FOK"

        vnpy_mock.trader.constant.Exchange = MockExchange
        vnpy_mock.trader.constant.Interval = MockInterval
        vnpy_mock.trader.constant.Direction = MockDirection
        vnpy_mock.trader.constant.Offset = MockOffset
        vnpy_mock.trader.constant.Status = MockStatus
        vnpy_mock.trader.constant.OrderType = MockOrderType

        # 让 qp.common.constants / strategies 能正常导入
        sys.modules["vnpy.trader.constant"] = MagicMock()
        sys.modules["vnpy.trader.constant"].Exchange = MockExchange
        sys.modules["vnpy.trader.constant"].Interval = MockInterval
        sys.modules["vnpy.trader.constant"].Direction = MockDirection
        sys.modules["vnpy.trader.constant"].Offset = MockOffset
        sys.modules["vnpy.trader.constant"].Status = MockStatus
        sys.modules["vnpy.trader.constant"].OrderType = MockOrderType

        # 创建真实的 BarData / TickData / OrderData / TradeData mock（保持属性赋值）
        class _RealBarData:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.vt_symbol = f"{kwargs.get('symbol','')}.{kwargs.get('exchange','').value if hasattr(kwargs.get('exchange',''), 'value') else kwargs.get('exchange','')}"

        class _RealTickData:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.vt_symbol = f"{kwargs.get('symbol','')}.{kwargs.get('exchange','').value if hasattr(kwargs.get('exchange',''), 'value') else kwargs.get('exchange','')}"

        class _RealOrderData:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.vt_symbol = f"{kwargs.get('symbol','')}.{kwargs.get('exchange','').value if hasattr(kwargs.get('exchange',''), 'value') else kwargs.get('exchange','')}"
                self.vt_orderid = kwargs.get('vt_orderid', kwargs.get('orderid', ''))
                if kwargs.get('gateway_name') and kwargs.get('orderid') and '.' not in self.vt_orderid:
                    self.vt_orderid = f"{kwargs.get('gateway_name')}.{kwargs.get('orderid')}"
            def is_active(self):
                status = getattr(self, 'status', None)
                return status in {MockStatus.SUBMITTING, MockStatus.NOTTRADED, MockStatus.PARTTRADED}

        class _RealTradeData:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.vt_symbol = f"{kwargs.get('symbol','')}.{kwargs.get('exchange','').value if hasattr(kwargs.get('exchange',''), 'value') else kwargs.get('exchange','')}"
                self.vt_orderid = kwargs.get('vt_orderid', kwargs.get('orderid', ''))
                self.vt_tradeid = kwargs.get('vt_tradeid', kwargs.get('tradeid', ''))

        obj_mock = MagicMock()
        obj_mock.BarData = _RealBarData
        obj_mock.TickData = _RealTickData
        obj_mock.OrderData = _RealOrderData
        obj_mock.TradeData = _RealTradeData
        sys.modules["vnpy.trader.object"] = obj_mock

        utility_mock = MagicMock()
        class _BarGenerator:
            def __init__(self, on_bar=None, *args, **kwargs):
                self.on_bar = on_bar
            def update_tick(self, tick):
                return None
        class _ArrayManager:
            def __init__(self, size=100):
                self.size = size
                self.inited = False
            def update_bar(self, bar):
                return None
        utility_mock.BarGenerator = _BarGenerator
        utility_mock.ArrayManager = _ArrayManager
        sys.modules["vnpy.trader.utility"] = utility_mock

        class _MockCtaTemplate:
            def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
                self.cta_engine = cta_engine
                self.strategy_name = strategy_name
                self.vt_symbol = vt_symbol
                self.inited = False
                self.trading = False
                self.pos = 0
                self._logs = []
                for k, v in setting.items():
                    setattr(self, k, v)
            def buy(self, price, volume, stop=False, lock=False, net=False):
                return self.send_order(MockDirection.LONG, MockOffset.OPEN, price, volume, stop, lock, net)
            def sell(self, price, volume, stop=False, lock=False, net=False):
                return self.send_order(MockDirection.SHORT, MockOffset.CLOSE, price, volume, stop, lock, net)
            def short(self, price, volume, stop=False, lock=False, net=False):
                return self.send_order(MockDirection.SHORT, MockOffset.OPEN, price, volume, stop, lock, net)
            def cover(self, price, volume, stop=False, lock=False, net=False):
                return self.send_order(MockDirection.LONG, MockOffset.CLOSE, price, volume, stop, lock, net)
            def send_order(self, direction, offset, price, volume, stop=False, lock=False, net=False):
                if self.trading and hasattr(self.cta_engine, 'send_order'):
                    return self.cta_engine.send_order(self, direction, offset, price, volume, stop, lock, net)
                return []
            def write_log(self, msg):
                self._logs.append(msg)
            def sync_data(self):
                return None
            def put_event(self):
                return None
            def load_bar(self, *args, **kwargs):
                return None

        cta_mock = MagicMock()
        cta_mock.CtaTemplate = _MockCtaTemplate
        sys.modules["vnpy_ctastrategy"] = cta_mock
        base_mock = MagicMock()
        class _MockEngineType(str, Enum):
            LIVE = 'live'
            BACKTESTING = 'backtesting'
        class _MockStopOrder:
            pass
        base_mock.EngineType = _MockEngineType
        base_mock.StopOrder = _MockStopOrder
        sys.modules["vnpy_ctastrategy.base"] = base_mock


_ensure_vnpy_mock()
