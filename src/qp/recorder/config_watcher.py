"""
配置热更新：事件驱动.

监听 data_recorder_setting.json 的文件变化，
检测到变化时 diff 新旧合约列表，回调通知 Headless Recorder 增减订阅。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from qp.common.logging import get_logger

from .constants import RECORDER_SETTING_FILE

logger = get_logger(__name__)

# 尝试使用 watchdog，不可用时 fallback 到 polling
try:
    from watchdog.events import FileModifiedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    logger.warning("watchdog 未安装，热更新将使用 polling fallback（5秒间隔）")


def load_setting(path: Path | None = None) -> dict:
    """加载 data_recorder_setting.json."""
    path = path or RECORDER_SETTING_FILE
    if not path.exists():
        return {"tick": {}, "bar": {}}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("读取配置文件失败: %s", e)
        return {"tick": {}, "bar": {}}


class ConfigChange:
    """配置变化描述."""

    def __init__(
        self,
        bar_added: set[str],
        bar_removed: set[str],
        tick_added: set[str],
        tick_removed: set[str],
        new_setting: dict,
    ) -> None:
        self.bar_added = bar_added
        self.bar_removed = bar_removed
        self.tick_added = tick_added
        self.tick_removed = tick_removed
        self.new_setting = new_setting

    @property
    def has_changes(self) -> bool:
        return bool(
            self.bar_added
            or self.bar_removed
            or self.tick_added
            or self.tick_removed
        )

    def __repr__(self) -> str:
        parts = []
        if self.bar_added:
            parts.append(f"bar+{self.bar_added}")
        if self.bar_removed:
            parts.append(f"bar-{self.bar_removed}")
        if self.tick_added:
            parts.append(f"tick+{self.tick_added}")
        if self.tick_removed:
            parts.append(f"tick-{self.tick_removed}")
        return f"ConfigChange({', '.join(parts) or 'no changes'})"


# 回调类型: (ConfigChange) -> None
OnConfigChange = Callable[[ConfigChange], None]


class ConfigWatcher:
    """配置文件热更新监视器.

    优先使用 watchdog 事件驱动，fallback 到 polling。

    用法:
        watcher = ConfigWatcher(on_change=my_callback)
        watcher.start()
        # ... 运行中 ...
        watcher.stop()
    """

    def __init__(
        self,
        on_change: OnConfigChange,
        setting_path: Path | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._on_change = on_change
        self._setting_path = setting_path or RECORDER_SETTING_FILE
        self._poll_interval = poll_interval

        # 当前状态
        self._current_setting = load_setting(self._setting_path)
        self._current_bars = set(self._current_setting.get("bar", {}).keys())
        self._current_ticks = set(self._current_setting.get("tick", {}).keys())

        self._stop_event = threading.Event()
        self._observer = None
        self._poll_thread: threading.Thread | None = None

    def _diff_and_notify(self) -> None:
        """重新加载配置并计算 diff."""
        new_setting = load_setting(self._setting_path)
        new_bars = set(new_setting.get("bar", {}).keys())
        new_ticks = set(new_setting.get("tick", {}).keys())

        change = ConfigChange(
            bar_added=new_bars - self._current_bars,
            bar_removed=self._current_bars - new_bars,
            tick_added=new_ticks - self._current_ticks,
            tick_removed=self._current_ticks - new_ticks,
            new_setting=new_setting,
        )

        self._current_setting = new_setting
        self._current_bars = new_bars
        self._current_ticks = new_ticks

        if change.has_changes:
            logger.info("检测到配置变化: %s", change)
            try:
                self._on_change(change)
            except Exception:
                logger.exception("配置变化回调执行失败")

    def start(self) -> None:
        """启动监视器."""
        self._stop_event.clear()

        if HAS_WATCHDOG:
            self._start_watchdog()
        else:
            self._start_polling()

    def _start_watchdog(self) -> None:
        """使用 watchdog 监听文件变化."""

        class _Handler(FileSystemEventHandler):
            def __init__(self, watcher: ConfigWatcher):
                self._watcher = watcher
                self._debounce_timer: threading.Timer | None = None

            def on_modified(self, event: FileModifiedEvent) -> None:
                src = Path(str(event.src_path))
                if src.name != self._watcher._setting_path.name:
                    return

                # debounce: 500ms 内多次修改只触发一次
                if self._debounce_timer is not None:
                    self._debounce_timer.cancel()
                self._debounce_timer = threading.Timer(
                    0.5, self._watcher._diff_and_notify
                )
                self._debounce_timer.daemon = True
                self._debounce_timer.start()

        handler = _Handler(self)
        self._observer = Observer()
        watch_dir = str(self._setting_path.parent)
        self._observer.schedule(handler, watch_dir, recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info(
            "配置热更新已启动（watchdog 模式），监听: %s", self._setting_path
        )

    def _start_polling(self) -> None:
        """Fallback: polling 模式."""

        def _poll_loop() -> None:
            last_mtime = (
                self._setting_path.stat().st_mtime
                if self._setting_path.exists()
                else 0
            )
            while not self._stop_event.is_set():
                self._stop_event.wait(self._poll_interval)
                if self._stop_event.is_set():
                    break
                try:
                    if self._setting_path.exists():
                        mtime = self._setting_path.stat().st_mtime
                        if mtime != last_mtime:
                            last_mtime = mtime
                            self._diff_and_notify()
                except OSError:
                    pass

        self._poll_thread = threading.Thread(target=_poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(
            "配置热更新已启动（polling 模式，间隔 %.1fs），监听: %s",
            self._poll_interval,
            self._setting_path,
        )

    def stop(self) -> None:
        """停止监视器."""
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
            self._poll_thread = None
        logger.info("配置热更新已停止")

    @property
    def current_bars(self) -> set[str]:
        """当前正在录制的 bar 合约列表."""
        return set(self._current_bars)

    @property
    def current_ticks(self) -> set[str]:
        """当前正在录制的 tick 合约列表."""
        return set(self._current_ticks)

    @property
    def current_setting(self) -> dict:
        """当前配置快照."""
        return dict(self._current_setting)
