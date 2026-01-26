# src/qp/utils/chan_debugger.py
"""
缠论策略Debug工具类.

用于实盘交易时的数据记录、日志输出和关键指标监控。
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChanDebugger:
    """
    缠论策略Debug工具类.

    功能:
    1. 创建带时间戳的debug目录
    2. 记录K线数据(1m/5m)到CSV文件
    3. 记录缠论关键数据(笔/中枢)
    4. 记录信号和交易
    5. 实时打印日志到控制台和文件

    使用方式:
        debugger = ChanDebugger("CtaChanPivot", enabled=True)
        debugger.log_kline_1m(bar_dict)
        debugger.log_bi(bi_point, k_lines)
        debugger.log_signal(signal_dict, reason="3B买点")
    """

    def __init__(
        self,
        strategy_name: str,
        base_dir: str = "data/debug",
        enabled: bool = True,
        log_level: str = "DEBUG",
        log_console: bool = True,
    ):
        """
        初始化Debugger.

        Args:
            strategy_name: 策略名称
            base_dir: debug根目录
            enabled: 是否启用debug
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            log_console: 是否输出到控制台
        """
        self.enabled = enabled
        self.strategy_name = strategy_name

        if not enabled:
            self.logger = logging.getLogger(f"ChanDebug_{strategy_name}")
            return

        # 创建带时间戳的debug目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{strategy_name}_{timestamp}"
        self.debug_dir = Path(base_dir) / self.session_id
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志系统
        self._init_logger(log_level, log_console)

        # 初始化CSV文件
        self._init_csv_files()

        # 统计数据
        self.stats = {
            "start_time": timestamp,
            "end_time": "",
            "total_bars_1m": 0,
            "total_bars_5m": 0,
            "total_bi": 0,
            "total_pivot": 0,
            "total_signals": 0,
            "total_trades": 0,
            "total_pnl": 0.0,
        }

        # 缓存最近状态(用于状态变化检测)
        self._last_bi_count = 0
        self._last_pivot_count = 0

        self.logger.info(f"{'='*60}")
        self.logger.info(f"ChanDebugger 初始化完成")
        self.logger.info(f"策略: {strategy_name}")
        self.logger.info(f"目录: {self.debug_dir}")
        self.logger.info(f"{'='*60}")

    def _init_logger(self, level: str, log_console: bool) -> None:
        """初始化日志系统."""
        self.logger = logging.getLogger(f"ChanDebug_{id(self)}")
        self.logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        self.logger.handlers.clear()
        self.logger.propagate = False

        # 日志格式
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # 文件Handler
        log_file = self.debug_dir / "strategy.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)

        # 控制台Handler
        if log_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)

    def _init_csv_files(self) -> None:
        """初始化CSV数据文件."""
        # 文件路径
        self.kline_1m_file = self.debug_dir / "kline_1m.csv"
        self.kline_5m_file = self.debug_dir / "kline_5m.csv"
        self.bi_file = self.debug_dir / "chan_bi.csv"
        self.pivot_file = self.debug_dir / "chan_pivot.csv"
        self.signal_file = self.debug_dir / "signals.csv"
        self.trade_file = self.debug_dir / "trades.csv"

        # 写入CSV头
        self._write_csv_header(self.kline_1m_file, [
            "datetime", "open", "high", "low", "close", "volume"
        ])
        self._write_csv_header(self.kline_5m_file, [
            "datetime", "open", "high", "low", "close", "volume",
            "diff_5m", "dea_5m", "macd_5m", "atr",
            "diff_15m", "dea_15m", "trend_15m"
        ])
        self._write_csv_header(self.bi_file, [
            "datetime", "bi_idx", "type", "price", "k_idx", "k_count"
        ])
        self._write_csv_header(self.pivot_file, [
            "datetime", "pivot_idx", "zg", "zd", "zz",
            "start_bi_idx", "end_bi_idx", "status"
        ])
        self._write_csv_header(self.signal_file, [
            "datetime", "signal_type", "direction", "trigger_price",
            "stop_price", "atr", "reason"
        ])
        self._write_csv_header(self.trade_file, [
            "datetime", "action", "price", "volume", "position",
            "pnl", "cum_pnl", "signal_type"
        ])

    def _write_csv_header(self, filepath: Path, headers: List[str]) -> None:
        """写入CSV文件头."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _append_csv(self, filepath: Path, row: List[Any]) -> None:
        """追加CSV数据行."""
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # =========================================================================
    # K线数据记录
    # =========================================================================

    def log_kline_1m(self, bar: Dict) -> None:
        """
        记录1分钟K线.

        Args:
            bar: K线数据字典 {datetime, open, high, low, close, volume}
        """
        if not self.enabled:
            return

        dt = bar.get('datetime', '')
        if hasattr(dt, 'strftime'):
            dt = dt.strftime("%Y-%m-%d %H:%M:%S")

        self._append_csv(self.kline_1m_file, [
            dt,
            bar.get('open', 0),
            bar.get('high', 0),
            bar.get('low', 0),
            bar.get('close', 0),
            bar.get('volume', 0)
        ])
        self.stats['total_bars_1m'] += 1

    def log_kline_5m(
        self,
        bar: Dict,
        diff_5m: float = 0,
        dea_5m: float = 0,
        atr: float = 0,
        diff_15m: float = 0,
        dea_15m: float = 0
    ) -> None:
        """
        记录5分钟K线及指标.

        Args:
            bar: K线数据字典
            diff_5m: 5分钟MACD DIFF
            dea_5m: 5分钟MACD DEA
            atr: ATR值
            diff_15m: 15分钟MACD DIFF
            dea_15m: 15分钟MACD DEA
        """
        if not self.enabled:
            return

        dt = bar.get('datetime', '')
        if hasattr(dt, 'strftime'):
            dt = dt.strftime("%Y-%m-%d %H:%M:%S")

        macd_5m = 2 * (diff_5m - dea_5m)
        trend_15m = "多" if diff_15m > dea_15m else "空"

        self._append_csv(self.kline_5m_file, [
            dt,
            bar.get('open', 0),
            bar.get('high', 0),
            bar.get('low', 0),
            bar.get('close', 0),
            bar.get('volume', 0),
            f"{diff_5m:.4f}",
            f"{dea_5m:.4f}",
            f"{macd_5m:.4f}",
            f"{atr:.2f}",
            f"{diff_15m:.4f}",
            f"{dea_15m:.4f}",
            trend_15m
        ])
        self.stats['total_bars_5m'] += 1

        # 实时打印MACD状态
        macd_status = "金叉" if diff_5m > dea_5m else "死叉"
        self.logger.info(
            f"[5M] {dt} | C={bar.get('close', 0):.0f} | "
            f"MACD={macd_5m:.2f}({macd_status}) | "
            f"ATR={atr:.1f} | 15M趋势={trend_15m}"
        )

    # =========================================================================
    # 缠论关键指标记录
    # =========================================================================

    def log_bi(
        self,
        bi: Dict,
        bi_idx: int,
        k_lines_count: int
    ) -> None:
        """
        记录新笔.

        Args:
            bi: 笔数据 {type, price, idx, data}
            bi_idx: 笔序号
            k_lines_count: 当前K线数量
        """
        if not self.enabled:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bi_type = bi.get('type', 'unknown')
        bi_type_cn = "顶分型" if bi_type == 'top' else "底分型"

        self._append_csv(self.bi_file, [
            now,
            bi_idx,
            bi_type,
            bi.get('price', 0),
            bi.get('idx', 0),
            k_lines_count
        ])
        self.stats['total_bi'] += 1

        # 醒目打印新笔
        self.logger.info(
            f"[笔] 新{bi_type_cn} | "
            f"价格={bi.get('price', 0):.0f} | "
            f"K线索引={bi.get('idx', 0)} | "
            f"总笔数={bi_idx}"
        )

    def log_pivot(
        self,
        pivot: Dict,
        pivot_idx: int,
        status: str = "forming"
    ) -> None:
        """
        记录中枢.

        Args:
            pivot: 中枢数据 {zg, zd, start_bi_idx, end_bi_idx}
            pivot_idx: 中枢序号
            status: 状态 (forming, confirmed, broken)
        """
        if not self.enabled:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        zg = pivot.get('zg', 0)
        zd = pivot.get('zd', 0)
        zz = (zg + zd) / 2  # 中枢中心

        self._append_csv(self.pivot_file, [
            now,
            pivot_idx,
            f"{zg:.2f}",
            f"{zd:.2f}",
            f"{zz:.2f}",
            pivot.get('start_bi_idx', 0),
            pivot.get('end_bi_idx', 0),
            status
        ])
        self.stats['total_pivot'] += 1

        # 醒目打印中枢
        range_pts = zg - zd
        self.logger.info(
            f"[中枢] ZG={zg:.0f} | ZD={zd:.0f} | "
            f"ZZ={zz:.0f} | 区间={range_pts:.0f}点"
        )

    def log_inclusion(
        self,
        before_high: float,
        before_low: float,
        after_high: float,
        after_low: float,
        direction: int
    ) -> None:
        """
        记录K线包含处理.

        Args:
            before_high: 处理前高点
            before_low: 处理前低点
            after_high: 处理后高点
            after_low: 处理后低点
            direction: 包含方向 (1=向上, -1=向下)
        """
        if not self.enabled:
            return

        dir_str = "向上" if direction == 1 else "向下"
        self.logger.debug(
            f"[包含] {dir_str}处理 | "
            f"前: H={before_high:.0f} L={before_low:.0f} | "
            f"后: H={after_high:.0f} L={after_low:.0f}"
        )

    # =========================================================================
    # 信号和交易记录
    # =========================================================================

    def log_signal(
        self,
        signal_type: str,
        direction: str,
        trigger_price: float,
        stop_price: float,
        atr: float = 0,
        reason: str = ""
    ) -> None:
        """
        记录交易信号.

        Args:
            signal_type: 信号类型 (3B, 3S, 2B, 2S)
            direction: 方向 (Buy, Sell)
            trigger_price: 触发价格
            stop_price: 止损价格
            atr: 当前ATR
            reason: 信号原因
        """
        if not self.enabled:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._append_csv(self.signal_file, [
            now,
            signal_type,
            direction,
            f"{trigger_price:.2f}",
            f"{stop_price:.2f}",
            f"{atr:.2f}",
            reason
        ])
        self.stats['total_signals'] += 1

        # 醒目打印信号
        direction_cn = "做多" if direction == 'Buy' else "做空"
        risk = abs(trigger_price - stop_price)
        risk_atr = risk / atr if atr > 0 else 0

        self.logger.warning(f"{'='*60}")
        self.logger.warning(f"[信号] {signal_type} {direction_cn}")
        self.logger.warning(f"  触发价: {trigger_price:.0f}")
        self.logger.warning(f"  止损价: {stop_price:.0f}")
        self.logger.warning(f"  风险: {risk:.0f}点 ({risk_atr:.1f}ATR)")
        self.logger.warning(f"  原因: {reason}")
        self.logger.warning(f"{'='*60}")

    def log_trade(
        self,
        action: str,
        price: float,
        volume: int,
        position: int,
        pnl: float = 0,
        signal_type: str = ""
    ) -> None:
        """
        记录交易执行.

        Args:
            action: 动作 (OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT)
            price: 成交价格
            volume: 成交数量
            position: 当前持仓
            pnl: 本次盈亏
            signal_type: 信号类型
        """
        if not self.enabled:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats['total_pnl'] += pnl

        self._append_csv(self.trade_file, [
            now,
            action,
            f"{price:.2f}",
            volume,
            position,
            f"{pnl:.2f}",
            f"{self.stats['total_pnl']:.2f}",
            signal_type
        ])

        if 'OPEN' in action:
            self.stats['total_trades'] += 1

        # 醒目打印交易
        action_cn = {
            'OPEN_LONG': '开多',
            'OPEN_SHORT': '开空',
            'CLOSE_LONG': '平多',
            'CLOSE_SHORT': '平空'
        }.get(action, action)

        self.logger.warning(
            f"[交易] {action_cn} @ {price:.0f} x {volume} | "
            f"持仓={position} | 盈亏={pnl:+.0f} | 累计={self.stats['total_pnl']:+.0f}"
        )

    # =========================================================================
    # 状态监控
    # =========================================================================

    def log_position_status(
        self,
        position: int,
        entry_price: float,
        stop_price: float,
        current_price: float,
        trailing_active: bool = False
    ) -> None:
        """
        记录持仓状态.

        Args:
            position: 持仓方向 (1=多, -1=空, 0=空仓)
            entry_price: 入场价
            stop_price: 止损价
            current_price: 当前价
            trailing_active: 移动止损是否激活
        """
        if not self.enabled or position == 0:
            return

        if position == 1:
            unrealized = current_price - entry_price
            pos_str = "多"
        else:
            unrealized = entry_price - current_price
            pos_str = "空"

        trail_str = "已激活" if trailing_active else "未激活"

        self.logger.info(
            f"[持仓] {pos_str}单 | 入场={entry_price:.0f} | "
            f"止损={stop_price:.0f} | 浮盈={unrealized:+.0f} | "
            f"移动止损={trail_str}"
        )

    def log_chan_state(
        self,
        k_lines_count: int,
        bi_count: int,
        pivot_count: int,
        bi_points: List[Dict] = None,
        pivots: List[Dict] = None
    ) -> None:
        """
        记录缠论状态摘要.

        Args:
            k_lines_count: 处理后K线数量
            bi_count: 笔数量
            pivot_count: 中枢数量
            bi_points: 笔端点列表(用于详细输出)
            pivots: 中枢列表(用于详细输出)
        """
        if not self.enabled:
            return

        # 检查是否有变化
        if bi_count == self._last_bi_count and pivot_count == self._last_pivot_count:
            return

        self._last_bi_count = bi_count
        self._last_pivot_count = pivot_count

        self.logger.info(
            f"[缠论] K线={k_lines_count} | 笔={bi_count} | 中枢={pivot_count}"
        )

        # 打印最近的笔
        if bi_points and len(bi_points) >= 2:
            recent_count = min(3, len(bi_points))
            for i in range(recent_count):
                bi = bi_points[-(recent_count - i)]
                bi_type = "顶" if bi['type'] == 'top' else "底"
                idx = len(bi_points) - recent_count + i + 1
                self.logger.info(f"  笔{idx}: {bi_type} @ {bi['price']:.0f}")

        # 打印当前中枢
        if pivots and len(pivots) > 0:
            pv = pivots[-1]
            self.logger.info(
                f"  当前中枢: [{pv['zd']:.0f}, {pv['zg']:.0f}]"
            )

    # =========================================================================
    # 配置和摘要
    # =========================================================================

    def save_config(self, config: Dict) -> None:
        """
        保存策略配置.

        Args:
            config: 配置字典
        """
        if not self.enabled:
            return

        config_file = self.debug_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            # 处理不可序列化的对象
            safe_config = {}
            for k, v in config.items():
                try:
                    json.dumps(v)
                    safe_config[k] = v
                except (TypeError, ValueError):
                    safe_config[k] = str(v)

            json.dump(safe_config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"配置已保存: {config_file}")

    def save_summary(self) -> None:
        """保存运行摘要."""
        if not self.enabled:
            return

        self.stats['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = self.debug_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"{'='*60}")
        self.logger.info("运行摘要:")
        self.logger.info(f"  1分钟K线: {self.stats['total_bars_1m']} 条")
        self.logger.info(f"  5分钟K线: {self.stats['total_bars_5m']} 条")
        self.logger.info(f"  笔: {self.stats['total_bi']} 个")
        self.logger.info(f"  中枢: {self.stats['total_pivot']} 个")
        self.logger.info(f"  信号: {self.stats['total_signals']} 个")
        self.logger.info(f"  交易: {self.stats['total_trades']} 笔")
        self.logger.info(f"  累计盈亏: {self.stats['total_pnl']:.0f}")
        self.logger.info(f"摘要已保存: {summary_file}")
        self.logger.info(f"{'='*60}")

    def close(self) -> None:
        """关闭debugger,保存摘要."""
        self.save_summary()
        # 关闭所有handler
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
