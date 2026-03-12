"""
CSV 落盘写入器.

- CsvBarSink: 1m bar → CSV（Wind 格式对齐）
- CsvTickSink: tick → CSV
- 按合约+交易日自动分文件
- 完整行写入 + flush，保证断电不丢已写数据
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

from qp.common.logging import get_logger

from .constants import (
    BAR_1M_DIR,
    BAR_CSV_COLUMNS,
    DT_FORMAT,
    TICK_CSV_COLUMNS,
    TICK_DIR,
    trading_date,
)

if TYPE_CHECKING:
    from vnpy.trader.object import BarData, TickData

logger = get_logger(__name__)


class _CsvWriter:
    """管理单个 CSV 文件的写入句柄."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self._file: TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None

    def _ensure_open(self) -> csv.DictWriter:
        """懒打开文件，首次写入时创建目录和文件头."""
        if self._writer is not None:
            return self._writer

        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0

        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)

        if write_header:
            self._writer.writeheader()
            self._file.flush()

        return self._writer

    def write_row(self, row: dict) -> None:
        """写入一行并立即 flush."""
        writer = self._ensure_open()
        writer.writerow(row)
        assert self._file is not None
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        """关闭文件句柄."""
        if self._file is not None:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
            except OSError:
                pass
            finally:
                self._file = None
                self._writer = None


class CsvBarSink:
    """1m Bar CSV 落盘器.

    每个合约按交易日自动分文件:
        data/recordings/bar_1m/{vt_symbol}/{trading_date}.csv
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or BAR_1M_DIR
        # {vt_symbol: {trading_date_str: _CsvWriter}}
        self._writers: dict[str, dict[str, _CsvWriter]] = {}

    def _get_writer(self, vt_symbol: str, dt: datetime) -> _CsvWriter:
        """获取或创建对应合约+交易日的 writer."""
        td = trading_date(dt)
        symbol_writers = self._writers.setdefault(vt_symbol, {})

        if td not in symbol_writers:
            path = self.base_dir / vt_symbol / f"{td}.csv"
            symbol_writers[td] = _CsvWriter(path, BAR_CSV_COLUMNS)

        return symbol_writers[td]

    def write_bar(self, bar: "BarData") -> None:
        """写入一条完整的 1m bar."""
        vt_symbol = f"{bar.symbol}.{bar.exchange.value}"
        writer = self._get_writer(vt_symbol, bar.datetime)

        row = {
            "datetime": bar.datetime.strftime(DT_FORMAT),
            "open": bar.open_price,
            "high": bar.high_price,
            "low": bar.low_price,
            "close": bar.close_price,
            "volume": bar.volume,
            "open_interest": bar.open_interest,
            "turnover": bar.turnover,
        }
        writer.write_row(row)

    def close_symbol(self, vt_symbol: str) -> None:
        """关闭指定合约的所有文件句柄."""
        writers = self._writers.pop(vt_symbol, {})
        for w in writers.values():
            w.close()

    def close_all(self) -> None:
        """关闭所有文件句柄."""
        for symbol_writers in self._writers.values():
            for w in symbol_writers.values():
                w.close()
        self._writers.clear()


class CsvTickSink:
    """Tick CSV 落盘器.

    每个合约按交易日自动分文件:
        data/recordings/tick/{vt_symbol}/{trading_date}.csv
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or TICK_DIR
        self._writers: dict[str, dict[str, _CsvWriter]] = {}

    def _get_writer(self, vt_symbol: str, dt: datetime) -> _CsvWriter:
        """获取或创建对应合约+交易日的 writer."""
        td = trading_date(dt)
        symbol_writers = self._writers.setdefault(vt_symbol, {})

        if td not in symbol_writers:
            path = self.base_dir / vt_symbol / f"{td}.csv"
            symbol_writers[td] = _CsvWriter(path, TICK_CSV_COLUMNS)

        return symbol_writers[td]

    def write_tick(self, tick: "TickData") -> None:
        """写入一条 tick."""
        vt_symbol = f"{tick.symbol}.{tick.exchange.value}"
        writer = self._get_writer(vt_symbol, tick.datetime)

        row = {
            "datetime": tick.datetime.strftime(DT_FORMAT),
            "last_price": tick.last_price,
            "volume": tick.volume,
            "turnover": tick.turnover,
            "open_interest": tick.open_interest,
            "last_volume": tick.last_volume,
            "limit_up": tick.limit_up,
            "limit_down": tick.limit_down,
            "open_price": tick.open_price,
            "high_price": tick.high_price,
            "low_price": tick.low_price,
            "pre_close": tick.pre_close,
            "bid_price_1": tick.bid_price_1,
            "ask_price_1": tick.ask_price_1,
            "bid_volume_1": tick.bid_volume_1,
            "ask_volume_1": tick.ask_volume_1,
        }

        # 五档（可能为 0 或 None）
        for i in range(2, 6):
            row[f"bid_price_{i}"] = getattr(tick, f"bid_price_{i}", 0) or 0
            row[f"ask_price_{i}"] = getattr(tick, f"ask_price_{i}", 0) or 0
            row[f"bid_volume_{i}"] = getattr(tick, f"bid_volume_{i}", 0) or 0
            row[f"ask_volume_{i}"] = getattr(tick, f"ask_volume_{i}", 0) or 0

        writer.write_row(row)

    def close_symbol(self, vt_symbol: str) -> None:
        """关闭指定合约的所有文件句柄."""
        writers = self._writers.pop(vt_symbol, {})
        for w in writers.values():
            w.close()

    def close_all(self) -> None:
        """关闭所有文件句柄."""
        for symbol_writers in self._writers.values():
            for w in symbol_writers.values():
                w.close()
        self._writers.clear()
