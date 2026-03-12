"""
Headless Recorder（无头采集器）主程序.

独立常驻运行，不依赖 GUI：
- 连接 CTP / TTS gateway
- 订阅合约行情（读取 data_recorder_setting.json）
- tick → 1m bar 聚合 → CSV 落盘
- tick → CSV 落盘
- 支持合约热更新（事件驱动）
- **不写数据库**

用法:
    # 直接运行
    python -m qp.recorder.headless

    # 指定 gateway（默认 CTP）
    python -m qp.recorder.headless --gateway tts

    # 指定配置文件路径
    python -m qp.recorder.headless --config .vntrader/data_recorder_setting.json
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from copy import copy
from datetime import datetime, timedelta
from logging import INFO
from pathlib import Path
from threading import Event as ThreadEvent

from qp.common.logging import get_logger, setup_logging

from .config_watcher import ConfigChange, ConfigWatcher, load_setting
from .constants import RECORDER_SETTING_FILE, REPO_ROOT
from .csv_sink import CsvBarSink, CsvTickSink

logger = get_logger(__name__)


class HeadlessRecorder:
    """无头采集器.

    核心职责：
    1. 连接 gateway，订阅合约
    2. 接收 tick，通过 BarGenerator 聚合 1m bar
    3. 完整 bar 写 CSV（CsvBarSink）
    4. tick 写 CSV（CsvTickSink）
    5. 响应配置热更新（增减订阅）
    """

    def __init__(
        self,
        gateway_name: str = "CTP",
        gateway_setting: dict | None = None,
        setting_path: Path | None = None,
    ) -> None:
        from vnpy.event import EventEngine, Event
        from vnpy.trader.engine import MainEngine
        from vnpy.trader.event import EVENT_TICK, EVENT_CONTRACT, EVENT_TIMER
        from vnpy.trader.object import ContractData, TickData, SubscribeRequest
        from vnpy.trader.utility import BarGenerator

        self._gateway_name = gateway_name.upper()
        self._gateway_instance_name = self._gateway_name
        self._gateway_setting = gateway_setting or {}
        self._setting_path = setting_path or RECORDER_SETTING_FILE

        # ── vnpy 引擎 ──
        self._event_engine = EventEngine()
        self._main_engine = MainEngine(self._event_engine)

        # ── CSV 落盘器 ──
        self._bar_sink = CsvBarSink()
        self._tick_sink = CsvTickSink()

        # ── 内部状态 ──
        self._bar_generators: dict[str, BarGenerator] = {}
        self._bar_symbols: set[str] = set()
        self._tick_symbols: set[str] = set()
        self._subscribed: set[str] = set()
        self._contracts: dict[str, ContractData] = {}

        # ── tick 时间过滤 ──
        self._filter_window = 60  # 秒
        self._filter_delta = timedelta(seconds=self._filter_window)

        # ── timer 批量落盘计数 ──
        self._timer_count = 0
        self._timer_interval = 10  # 每 10 秒 flush 一次（safety net）

        # ── 配置热更新 ──
        self._config_watcher = ConfigWatcher(
            on_change=self._on_config_change,
            setting_path=self._setting_path,
        )

        # ── 统计计数 ──
        self._tick_count: dict[str, int] = {}  # vt_symbol -> tick count
        self._bar_count: dict[str, int] = {}   # vt_symbol -> bar count
        self._first_tick_logged: set[str] = set()  # 已记录首笔 tick 的合约
        self._status_interval = 300  # 每 300 秒打印一次状态摘要
        self._status_timer = 0

        # ── 事件注册 ──
        self._event_engine.register(EVENT_TICK, self._on_tick_event)
        self._event_engine.register(EVENT_CONTRACT, self._on_contract_event)
        self._event_engine.register(EVENT_TIMER, self._on_timer_event)

        # ── 账户事件 ──
        from vnpy.trader.event import EVENT_ACCOUNT
        self._event_engine.register(EVENT_ACCOUNT, self._on_account_event)

        # ── 停止信号 ──
        self._stop_event = ThreadEvent()

    def _load_initial_config(self) -> None:
        """加载初始配置."""
        setting = load_setting(self._setting_path)
        self._bar_symbols = set(setting.get("bar", {}).keys())
        self._tick_symbols = set(setting.get("tick", {}).keys())
        self._filter_window = setting.get("filter_window", 60)
        self._filter_delta = timedelta(seconds=self._filter_window)

        logger.info(
            "初始配置: bar %d 个合约, tick %d 个合约",
            len(self._bar_symbols),
            len(self._tick_symbols),
        )
        if self._bar_symbols:
            logger.info("  Bar 合约: %s", ", ".join(sorted(self._bar_symbols)))
        if self._tick_symbols:
            logger.info("  Tick 合约: %s", ", ".join(sorted(self._tick_symbols)))

    def _on_config_change(self, change: ConfigChange) -> None:
        """处理配置热更新回调."""
        from vnpy.trader.utility import BarGenerator

        # Bar 合约变化
        for vt_symbol in change.bar_added:
            self._bar_symbols.add(vt_symbol)
            logger.info("热更新: 添加 bar 录制 %s", vt_symbol)
            self._try_subscribe(vt_symbol)

        for vt_symbol in change.bar_removed:
            self._bar_symbols.discard(vt_symbol)
            self._bar_generators.pop(vt_symbol, None)
            self._bar_sink.close_symbol(vt_symbol)
            logger.info("热更新: 停止 bar 录制 %s", vt_symbol)

        # Tick 合约变化
        for vt_symbol in change.tick_added:
            self._tick_symbols.add(vt_symbol)
            logger.info("热更新: 添加 tick 录制 %s", vt_symbol)
            self._try_subscribe(vt_symbol)

        for vt_symbol in change.tick_removed:
            self._tick_symbols.discard(vt_symbol)
            self._tick_sink.close_symbol(vt_symbol)
            logger.info("热更新: 停止 tick 录制 %s", vt_symbol)

    def _try_subscribe(self, vt_symbol: str) -> None:
        """尝试订阅合约行情."""
        from vnpy.trader.object import SubscribeRequest

        if vt_symbol in self._subscribed:
            return

        contract = self._contracts.get(vt_symbol)
        if contract:
            req = SubscribeRequest(
                symbol=contract.symbol,
                exchange=contract.exchange,
            )
            self._main_engine.subscribe(req, contract.gateway_name)
            self._subscribed.add(vt_symbol)
            logger.info("订阅行情: %s (gateway: %s)", vt_symbol, contract.gateway_name)

    def _on_log_event(self, event) -> None:
        """捕获 vnpy gateway 内部日志（连接状态、认证、错误等）."""
        log = event.data
        msg = getattr(log, "msg", str(log))
        level = getattr(log, "level", 20)  # INFO=20
        gateway = getattr(log, "gateway_name", "")
        prefix = f"[{gateway}] " if gateway else ""
        logger.log(level, "%s%s", prefix, msg)

    def _on_account_event(self, event) -> None:
        """捕获账户信息（资金、可用余额等）."""
        account = event.data
        logger.info(
            "[账户] %s | 余额: %.2f | 冻结: %.2f | 可用: %.2f",
            getattr(account, "accountid", "?"),
            getattr(account, "balance", 0),
            getattr(account, "frozen", 0),
            getattr(account, "available", 0),
        )

    def _on_contract_event(self, event) -> None:
        """收到合约信息推送，缓存并尝试订阅."""
        contract = event.data
        vt_symbol = contract.vt_symbol
        self._contracts[vt_symbol] = contract

        if vt_symbol in self._bar_symbols or vt_symbol in self._tick_symbols:
            self._try_subscribe(vt_symbol)

    def _on_tick_event(self, event) -> None:
        """收到 tick 推送."""
        from vnpy.trader.utility import BarGenerator

        tick = event.data
        vt_symbol = tick.vt_symbol

        # 时间过滤
        now = datetime.now(tick.datetime.tzinfo) if tick.datetime.tzinfo else datetime.now()
        delta = abs(tick.datetime - now)
        if delta >= self._filter_delta:
            return

        # 首笔 tick 日志
        if vt_symbol not in self._first_tick_logged:
            self._first_tick_logged.add(vt_symbol)
            logger.info(
                "✅ 收到首笔 tick: %s | price=%.2f vol=%d oi=%d | %s",
                vt_symbol,
                tick.last_price,
                tick.volume,
                tick.open_interest,
                tick.datetime.strftime("%H:%M:%S"),
            )

        # Tick 计数
        self._tick_count[vt_symbol] = self._tick_count.get(vt_symbol, 0) + 1

        # Tick 录制
        if vt_symbol in self._tick_symbols:
            self._tick_sink.write_tick(copy(tick))

        # Bar 录制（聚合）
        if vt_symbol in self._bar_symbols:
            bg = self._bar_generators.get(vt_symbol)
            if bg is None:
                bg = BarGenerator(
                    on_bar=lambda bar, _vt=vt_symbol: self._on_bar(bar),
                )
                self._bar_generators[vt_symbol] = bg
            bg.update_tick(copy(tick))

    def _on_bar(self, bar) -> None:
        """BarGenerator 完成一根 1m bar 的回调."""
        vt_symbol = f"{bar.symbol}.{bar.exchange.value}"
        self._bar_count[vt_symbol] = self._bar_count.get(vt_symbol, 0) + 1
        count = self._bar_count[vt_symbol]

        self._bar_sink.write_bar(bar)

        # 前 3 根 bar 详细打印，之后每 10 根打印一次
        if count <= 3 or count % 10 == 0:
            logger.info(
                "📊 Bar #%d: %s | %s | O=%.1f H=%.1f L=%.1f C=%.1f V=%d",
                count,
                vt_symbol,
                bar.datetime.strftime("%H:%M"),
                bar.open_price,
                bar.high_price,
                bar.low_price,
                bar.close_price,
                bar.volume,
            )

    def _on_timer_event(self, event) -> None:
        """定时器事件：定期输出状态摘要."""
        self._timer_count += 1
        self._status_timer += 1

        if self._timer_count < self._timer_interval:
            return
        self._timer_count = 0

        # 每 _status_interval 秒打印一次状态摘要
        if self._status_timer >= self._status_interval:
            self._status_timer = 0
            self._print_status()

    def _print_status(self) -> None:
        """打印当前采集状态摘要."""
        total_ticks = sum(self._tick_count.values())
        total_bars = sum(self._bar_count.values())
        subscribed = len(self._subscribed)
        pending_bar = len(self._bar_symbols - self._subscribed)
        pending_tick = len(self._tick_symbols - self._subscribed)

        logger.info("=" * 50)
        logger.info("📈 状态摘要")
        logger.info("  已订阅合约: %d", subscribed)
        if pending_bar or pending_tick:
            logger.info("  待订阅: bar %d, tick %d", pending_bar, pending_tick)
        logger.info("  累计 Tick: %d | 累计 Bar: %d", total_ticks, total_bars)

        for vt in sorted(self._bar_symbols | self._tick_symbols):
            tc = self._tick_count.get(vt, 0)
            bc = self._bar_count.get(vt, 0)
            status = "✅" if vt in self._subscribed else "⏳"
            logger.info("  %s %s: tick=%d bar=%d", status, vt, tc, bc)
        logger.info("=" * 50)

    def _add_gateway(self) -> None:
        """添加 md-only gateway（只连行情，不连交易）."""
        from .md_only_gateway import get_md_only_gateway_class

        gw_name = self._gateway_name.upper()
        gateway_cls = get_md_only_gateway_class(gw_name)
        self._gateway_instance_name = gateway_cls.default_name
        self._main_engine.add_gateway(gateway_cls, self._gateway_instance_name)

    def start(self) -> None:
        """启动无头采集器."""
        logger.info("=" * 60)
        logger.info("Headless Recorder 启动")
        logger.info("Gateway 类型: %s", self._gateway_name)
        logger.info("配置文件: %s", self._setting_path)
        logger.info("=" * 60)

        # 1. 加载初始配置
        self._load_initial_config()

        if not self._bar_symbols and not self._tick_symbols:
            logger.warning("配置为空，没有合约需要录制。等待热更新...")

        # 2. 添加 gateway
        self._add_gateway()

        # 3. 启动配置热更新
        self._config_watcher.start()

        # 4. 连接 gateway
        connect_setting = self._gateway_setting
        connect_source = "命令行参数"

        if not connect_setting:
            # 尝试使用已保存的连接配置
            from vnpy.trader.utility import load_json
            connect_file = f"connect_{self._gateway_name.lower()}.json"
            connect_setting = load_json(connect_file)
            connect_source = connect_file

        if not connect_setting:
            logger.error(
                "找不到 %s 连接配置。请先在 GUI 中连接一次以生成配置文件，"
                "或手动创建 .vntrader/connect_%s.json",
                self._gateway_name,
                self._gateway_name.lower(),
            )
            return

        # 注入录制合约，供 md-only gateway 预注册 ContractData
        connect_setting = dict(connect_setting)
        connect_setting["_record_symbols"] = sorted(self._bar_symbols | self._tick_symbols)

        # 打印连接参数（密码脱敏）
        self._log_connect_info(connect_setting, connect_source)

        self._main_engine.connect(connect_setting, self._gateway_instance_name)
        logger.info("正在连接 %s (%s)...", self._gateway_instance_name, self._gateway_name)
        logger.info("模式: Md-only（仅行情，不连接交易服务器）")
        logger.info("等待行情登录与首笔 tick（通常需要 5~15 秒）...")

    def wait(self) -> None:
        """阻塞等待直到收到停止信号."""
        logger.info("Headless Recorder 正在运行，Ctrl+C 停止")
        try:
            # Windows 上 Event.wait() 无超时时会吃掉 KeyboardInterrupt
            # 用短超时循环让 Python 有机会处理信号
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """停止无头采集器."""
        logger.info("正在停止 Headless Recorder...")

        self._stop_event.set()
        self._config_watcher.stop()
        self._bar_sink.close_all()
        self._tick_sink.close_all()
        self._main_engine.close()

        logger.info("Headless Recorder 已停止")

    def _log_connect_info(self, setting: dict, source: str) -> None:
        """打印连接参数（密码脱敏）."""
        logger.info("连接配置来源: %s", source)

        # CTP/TTS 常见字段名（中英文都处理）
        user_keys = ["用户名", "userid", "user", "username"]
        broker_keys = ["经纪商代码", "brokerid", "broker"]
        td_keys = ["交易服务器", "td_address", "trade_server"]
        md_keys = ["行情服务器", "md_address", "market_server"]
        auth_keys = ["产品名称", "appid", "app_id", "product_name"]

        def _find(keys: list[str]) -> str:
            for k in keys:
                for sk in setting:
                    if sk.lower() == k.lower():
                        return str(setting[sk])
            return "未配置"

        user = _find(user_keys)
        broker = _find(broker_keys)
        td = _find(td_keys)
        md = _find(md_keys)
        auth = _find(auth_keys)

        logger.info("  用户: %s", user)
        logger.info("  经纪商: %s", broker)
        logger.info("  行情服务器: %s", md)
        logger.info("  交易服务器: %s（Md-only 模式忽略）", td)
        logger.info("  产品名称: %s", auth)
        # 密码绝不打印

    def request_stop(self) -> None:
        """从外部请求停止（用于信号处理）."""
        self._stop_event.set()


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        prog="qp.recorder.headless",
        description="Headless Recorder - 无头行情采集器",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        default="CTP",
        choices=["CTP", "TTS", "ctp", "tts"],
        help="Gateway 类型 (默认: CTP)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: .vntrader/data_recorder_setting.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 切到仓库根目录（vnpy 需要）
    os.chdir(REPO_ROOT)

    setup_logging(verbose=args.verbose)

    gateway_name = args.gateway.upper()
    setting_path = Path(args.config) if args.config else None

    # 不在 main 里预加载连接参数，让 start() 自己从 connect_*.json 读取
    # 这样日志里的“连接配置来源”会显示正确文件名，而不是误报“命令行参数”
    recorder = HeadlessRecorder(
        gateway_name=gateway_name,
        gateway_setting=None,
        setting_path=setting_path,
    )

    # 信号处理
    def _signal_handler(sig, frame):
        logger.info("收到信号 %s，准备停止...", sig)
        recorder.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    recorder.start()
    recorder.wait()


if __name__ == "__main__":
    main()
