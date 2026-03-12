"""
Recorder 模块单元测试.

测试 CSV 落盘、配置热更新、数据库同步的核心逻辑。
不依赖 vnpy 运行时（通过 mock）。
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Constants ──────────────────────────────────────────────


class TestTradingDate:
    """测试交易日期计算."""

    def test_day_session(self):
        from qp.recorder.constants import trading_date

        dt = datetime(2026, 3, 9, 10, 30, 0)
        assert trading_date(dt) == "2026-03-09"

    def test_night_session(self):
        from qp.recorder.constants import trading_date

        dt = datetime(2026, 3, 9, 21, 1, 0)
        assert trading_date(dt) == "2026-03-10"

    def test_night_session_boundary(self):
        from qp.recorder.constants import trading_date

        dt = datetime(2026, 3, 9, 21, 0, 0)
        assert trading_date(dt) == "2026-03-10"

    def test_early_morning(self):
        from qp.recorder.constants import trading_date

        dt = datetime(2026, 3, 10, 0, 30, 0)
        assert trading_date(dt) == "2026-03-10"

    def test_afternoon_close(self):
        from qp.recorder.constants import trading_date

        dt = datetime(2026, 3, 9, 15, 0, 0)
        assert trading_date(dt) == "2026-03-09"


# ── CSV Sink ───────────────────────────────────────────────


class FakeBarData:
    """模拟 vnpy BarData."""

    def __init__(self, symbol, exchange_value, dt, o, h, l, c, v, oi, turnover):
        self.symbol = symbol
        self.exchange = MagicMock()
        self.exchange.value = exchange_value
        self.datetime = dt
        self.open_price = o
        self.high_price = h
        self.low_price = l
        self.close_price = c
        self.volume = v
        self.open_interest = oi
        self.turnover = turnover


class FakeTickData:
    """模拟 vnpy TickData."""

    def __init__(self, symbol, exchange_value, dt):
        self.symbol = symbol
        self.exchange = MagicMock()
        self.exchange.value = exchange_value
        self.datetime = dt
        self.last_price = 8500.0
        self.volume = 100
        self.turnover = 850000.0
        self.open_interest = 50000
        self.last_volume = 10
        self.limit_up = 9000.0
        self.limit_down = 8000.0
        self.open_price = 8450.0
        self.high_price = 8550.0
        self.low_price = 8400.0
        self.pre_close = 8430.0
        self.bid_price_1 = 8499.0
        self.ask_price_1 = 8501.0
        self.bid_volume_1 = 50
        self.ask_volume_1 = 30
        # 2-5 档
        for i in range(2, 6):
            setattr(self, f"bid_price_{i}", 8499.0 - i)
            setattr(self, f"ask_price_{i}", 8501.0 + i)
            setattr(self, f"bid_volume_{i}", 50 - i * 5)
            setattr(self, f"ask_volume_{i}", 30 - i * 3)


class TestCsvBarSink:
    """测试 Bar CSV 写入."""

    def test_write_bar_creates_file(self, tmp_path):
        from qp.recorder.csv_sink import CsvBarSink

        sink = CsvBarSink(base_dir=tmp_path)
        bar = FakeBarData(
            "p2605", "DCE",
            datetime(2026, 3, 9, 10, 1, 0),
            8438.0, 8456.0, 8426.0, 8440.0,
            2129, 137493, 179730520.0,
        )

        sink.write_bar(bar)
        sink.close_all()

        csv_path = tmp_path / "p2605.DCE" / "2026-03-09.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["datetime"] == "2026-03-09 10:01:00"
        assert float(rows[0]["open"]) == 8438.0
        assert float(rows[0]["close"]) == 8440.0
        assert float(rows[0]["volume"]) == 2129.0

    def test_write_bar_night_session_goes_to_next_day(self, tmp_path):
        from qp.recorder.csv_sink import CsvBarSink

        sink = CsvBarSink(base_dir=tmp_path)
        bar = FakeBarData(
            "p2605", "DCE",
            datetime(2026, 3, 9, 21, 1, 0),
            8438.0, 8456.0, 8426.0, 8440.0,
            500, 130000, 42300000.0,
        )

        sink.write_bar(bar)
        sink.close_all()

        # 夜盘归属次日
        csv_path = tmp_path / "p2605.DCE" / "2026-03-10.csv"
        assert csv_path.exists()

    def test_write_multiple_bars_appends(self, tmp_path):
        from qp.recorder.csv_sink import CsvBarSink

        sink = CsvBarSink(base_dir=tmp_path)

        for minute in range(1, 4):
            bar = FakeBarData(
                "p2605", "DCE",
                datetime(2026, 3, 9, 10, minute, 0),
                8438.0 + minute, 8456.0, 8426.0, 8440.0 + minute,
                2000 + minute * 100, 137000, 170000000.0,
            )
            sink.write_bar(bar)

        sink.close_all()

        csv_path = tmp_path / "p2605.DCE" / "2026-03-09.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]["datetime"] == "2026-03-09 10:01:00"
        assert rows[2]["datetime"] == "2026-03-09 10:03:00"

    def test_csv_columns_match_wind_format(self, tmp_path):
        """验证列名和 Wind 数据格式一致."""
        from qp.recorder.csv_sink import CsvBarSink
        from qp.recorder.constants import BAR_CSV_COLUMNS

        sink = CsvBarSink(base_dir=tmp_path)
        bar = FakeBarData(
            "p2605", "DCE",
            datetime(2026, 3, 9, 10, 1, 0),
            8438.0, 8456.0, 8426.0, 8440.0,
            2129, 137493, 179730520.0,
        )
        sink.write_bar(bar)
        sink.close_all()

        csv_path = tmp_path / "p2605.DCE" / "2026-03-09.csv"
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == BAR_CSV_COLUMNS


class TestCsvTickSink:
    """测试 Tick CSV 写入."""

    def test_write_tick_creates_file(self, tmp_path):
        from qp.recorder.csv_sink import CsvTickSink

        sink = CsvTickSink(base_dir=tmp_path)
        tick = FakeTickData(
            "p2605", "DCE",
            datetime(2026, 3, 9, 10, 0, 5),
        )

        sink.write_tick(tick)
        sink.close_all()

        csv_path = tmp_path / "p2605.DCE" / "2026-03-09.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["datetime"] == "2026-03-09 10:00:05"
        assert float(rows[0]["last_price"]) == 8500.0


# ── Config Watcher ─────────────────────────────────────────


class TestConfigWatcher:
    """测试配置热更新."""

    def test_load_setting(self, tmp_path):
        from qp.recorder.config_watcher import load_setting

        setting = {
            "tick": {"p2605.DCE": {"symbol": "p2605"}},
            "bar": {"p2605.DCE": {"symbol": "p2605"}},
        }
        setting_file = tmp_path / "data_recorder_setting.json"
        with open(setting_file, "w") as f:
            json.dump(setting, f)

        result = load_setting(setting_file)
        assert "p2605.DCE" in result["bar"]
        assert "p2605.DCE" in result["tick"]

    def test_load_setting_missing_file(self, tmp_path):
        from qp.recorder.config_watcher import load_setting

        result = load_setting(tmp_path / "nonexistent.json")
        assert result == {"tick": {}, "bar": {}}

    def test_config_change_detection(self, tmp_path):
        from qp.recorder.config_watcher import ConfigChange

        change = ConfigChange(
            bar_added={"IF2602.CFFEX"},
            bar_removed=set(),
            tick_added=set(),
            tick_removed={"p2509.DCE"},
            new_setting={},
        )

        assert change.has_changes
        assert "IF2602.CFFEX" in change.bar_added
        assert "p2509.DCE" in change.tick_removed

    def test_config_no_change(self):
        from qp.recorder.config_watcher import ConfigChange

        change = ConfigChange(
            bar_added=set(),
            bar_removed=set(),
            tick_added=set(),
            tick_removed=set(),
            new_setting={},
        )
        assert not change.has_changes

    def test_watcher_detects_file_change(self, tmp_path):
        """测试 watcher 能检测到文件变化（polling 模式）."""
        from qp.recorder.config_watcher import ConfigWatcher

        setting_file = tmp_path / "data_recorder_setting.json"

        # 初始配置
        initial = {"bar": {"p2605.DCE": {}}, "tick": {}}
        with open(setting_file, "w") as f:
            json.dump(initial, f)

        changes = []

        def on_change(change):
            changes.append(change)

        watcher = ConfigWatcher(
            on_change=on_change,
            setting_path=setting_file,
            poll_interval=0.5,  # 快速轮询用于测试
        )
        watcher.start()

        try:
            # 修改配置（添加合约）
            time.sleep(1)
            updated = {"bar": {"p2605.DCE": {}, "IF2602.CFFEX": {}}, "tick": {}}
            with open(setting_file, "w") as f:
                json.dump(updated, f)

            # 等待检测
            time.sleep(2)
        finally:
            watcher.stop()

        assert len(changes) >= 1
        assert "IF2602.CFFEX" in changes[0].bar_added


# ── DB Sync ────────────────────────────────────────────────


class TestDbSync:
    """测试数据库同步逻辑（不依赖真实 vnpy 数据库）."""

    def test_parse_vt_symbol(self):
        from qp.recorder.db_sync import _parse_vt_symbol

        assert _parse_vt_symbol("p2605.DCE") == ("p2605", "DCE")
        assert _parse_vt_symbol("IF2602.CFFEX") == ("IF2602", "CFFEX")

    def test_parse_vt_symbol_invalid(self):
        from qp.recorder.db_sync import _parse_vt_symbol

        with pytest.raises(ValueError):
            _parse_vt_symbol("invalid")

    def test_sync_state_roundtrip(self, tmp_path):
        from qp.recorder.db_sync import _load_sync_state, _save_sync_state
        from qp.recorder import constants

        # 临时替换路径
        orig = constants.SYNC_STATE_FILE
        constants.SYNC_STATE_FILE = tmp_path / "state.json"

        try:
            state = {"bar_1m/p2605.DCE/2026-03-09.csv": "2026-03-09 14:59:00"}
            _save_sync_state(state)

            loaded = _load_sync_state()
            assert loaded == state
        finally:
            constants.SYNC_STATE_FILE = orig

    def test_csv_to_bars_incremental(self, tmp_path):
        """测试增量 CSV → BarData 转换."""
        # 创建测试 CSV
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "datetime", "open", "high", "low", "close",
                "volume", "open_interest", "turnover",
            ])
            writer.writeheader()
            for i in range(5):
                writer.writerow({
                    "datetime": f"2026-03-09 10:0{i+1}:00",
                    "open": 8438.0 + i,
                    "high": 8460.0,
                    "low": 8420.0,
                    "close": 8440.0 + i,
                    "volume": 2000 + i * 100,
                    "open_interest": 137000,
                    "turnover": 170000000.0,
                })

        from qp.recorder.db_sync import _csv_to_bars_incremental

        # 全量
        bars = _csv_to_bars_incremental(csv_path, "p2605", "DCE")
        assert len(bars) == 5

        # 增量：只取 10:03 之后的
        bars = _csv_to_bars_incremental(
            csv_path, "p2605", "DCE",
            after_dt="2026-03-09 10:03:00",
        )
        assert len(bars) == 2
        assert bars[0].datetime == datetime(2026, 3, 9, 10, 4, 0)


# ── 集成测试 ───────────────────────────────────────────────


class TestIntegration:
    """端到端集成测试：写 CSV → 同步到 DB."""

    def test_bar_sink_then_sync(self, tmp_path):
        """写几根 bar 到 CSV，然后同步到 mock DB."""
        from qp.recorder.csv_sink import CsvBarSink
        from qp.recorder.db_sync import _csv_to_bars_incremental

        # 1. 写 CSV
        bar_dir = tmp_path / "bar_1m"
        sink = CsvBarSink(base_dir=bar_dir)

        for minute in range(1, 6):
            bar = FakeBarData(
                "p2605", "DCE",
                datetime(2026, 3, 9, 10, minute, 0),
                8438.0 + minute, 8460.0, 8420.0, 8440.0 + minute,
                2000 + minute * 100, 137000, 170000000.0,
            )
            sink.write_bar(bar)
        sink.close_all()

        # 2. 验证 CSV 存在
        csv_path = bar_dir / "p2605.DCE" / "2026-03-09.csv"
        assert csv_path.exists()

        # 3. 读取并验证
        bars = _csv_to_bars_incremental(csv_path, "p2605", "DCE")
        assert len(bars) == 5
        assert bars[0].open_price == 8439.0
        assert bars[4].close_price == 8445.0
