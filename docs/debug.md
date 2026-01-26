# 缠论策略 Debug 功能设计方案

## 一、概述

本方案为缠论策略(CtaChanPivotStrategy)设计一套完整的Debug功能，用于实盘交易时的数据记录、日志输出和关键指标监控。

## 二、目录结构

```
data/
└── debug/
    └── {strategy_name}_{YYYYMMDD_HHMMSS}/
        ├── config.json          # 策略启动配置
        ├── kline_1m.csv         # 1分钟K线数据
        ├── kline_5m.csv         # 5分钟合成K线
        ├── chan_bi.csv          # 笔数据记录
        ├── chan_pivot.csv       # 中枢数据记录
        ├── signals.csv          # 信号记录
        ├── trades.csv           # 交易记录
        ├── strategy.log         # 详细策略日志
        └── summary.json         # 运行摘要
```

## 三、核心类设计

### 3.1 ChanDebugger 类

```python
from datetime import datetime
from pathlib import Path
import json
import csv
import logging
from typing import Optional, Dict, List, Any

class ChanDebugger:
    """缠论策略Debug工具类"""

    def __init__(
        self,
        strategy_name: str,
        base_dir: str = "data/debug",
        enabled: bool = True,
        log_level: str = "DEBUG"
    ):
        self.enabled = enabled
        if not enabled:
            return

        # 创建带时间戳的debug目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = Path(base_dir) / f"{strategy_name}_{timestamp}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志
        self._init_logger(log_level)

        # 初始化CSV文件
        self._init_csv_files()

        # 统计数据
        self.stats = {
            "start_time": timestamp,
            "total_bars": 0,
            "total_bi": 0,
            "total_pivot": 0,
            "total_signals": 0,
            "total_trades": 0
        }

    def _init_logger(self, level: str):
        """初始化日志系统"""
        self.logger = logging.getLogger(f"ChanDebug_{id(self)}")
        self.logger.setLevel(getattr(logging, level))
        self.logger.handlers.clear()

        # 文件Handler
        fh = logging.FileHandler(
            self.debug_dir / "strategy.log",
            encoding='utf-8'
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(fh)

        # 控制台Handler (实时打印)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(ch)

    def _init_csv_files(self):
        """初始化CSV数据文件"""
        # K线文件
        self.kline_1m_file = self.debug_dir / "kline_1m.csv"
        self.kline_5m_file = self.debug_dir / "kline_5m.csv"

        # 缠论数据文件
        self.bi_file = self.debug_dir / "chan_bi.csv"
        self.pivot_file = self.debug_dir / "chan_pivot.csv"
        self.signal_file = self.debug_dir / "signals.csv"
        self.trade_file = self.debug_dir / "trades.csv"

        # 写入CSV头
        self._write_csv_header(self.kline_1m_file,
            ["datetime", "open", "high", "low", "close", "volume"])
        self._write_csv_header(self.kline_5m_file,
            ["datetime", "open", "high", "low", "close", "volume",
             "diff", "dea", "macd", "atr"])
        self._write_csv_header(self.bi_file,
            ["datetime", "type", "price", "idx", "k_count"])
        self._write_csv_header(self.pivot_file,
            ["datetime", "zg", "zd", "zz", "bi_start", "bi_end", "status"])
        self._write_csv_header(self.signal_file,
            ["datetime", "signal_type", "direction", "trigger_price",
             "stop_price", "reason"])
        self._write_csv_header(self.trade_file,
            ["datetime", "action", "price", "position", "pnl", "signal_type"])

    def _write_csv_header(self, filepath: Path, headers: List[str]):
        """写入CSV文件头"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _append_csv(self, filepath: Path, row: List[Any]):
        """追加CSV数据行"""
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
```

### 3.2 K线数据记录

```python
    def log_kline_1m(self, bar: Dict):
        """记录1分钟K线"""
        if not self.enabled:
            return

        self._append_csv(self.kline_1m_file, [
            bar['datetime'], bar['open'], bar['high'],
            bar['low'], bar['close'], bar['volume']
        ])
        self.stats['total_bars'] += 1

    def log_kline_5m(self, bar: Dict):
        """记录5分钟合成K线"""
        if not self.enabled:
            return

        self._append_csv(self.kline_5m_file, [
            bar['datetime'], bar['open'], bar['high'],
            bar['low'], bar['close'], bar['volume'],
            bar.get('diff', 0), bar.get('dea', 0),
            bar.get('macd', 0), bar.get('atr', 0)
        ])

        # 实时打印MACD状态
        macd_status = "金叉" if bar.get('diff', 0) > bar.get('dea', 0) else "死叉"
        self.logger.info(
            f"[5M] {bar['datetime']} | "
            f"C={bar['close']:.0f} | "
            f"MACD={bar.get('macd', 0):.2f} ({macd_status}) | "
            f"ATR={bar.get('atr', 0):.2f}"
        )
```

### 3.3 缠论关键指标记录

```python
    def log_bi(self, bi: Dict, k_lines: List):
        """记录笔"""
        if not self.enabled:
            return

        self._append_csv(self.bi_file, [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            bi['type'],  # 'top' or 'bot'
            bi['price'],
            bi['idx'],
            len(k_lines)
        ])
        self.stats['total_bi'] += 1

        # 实时打印
        bi_type = "顶分型" if bi['type'] == 'top' else "底分型"
        self.logger.info(
            f"[笔] 新{bi_type} | 价格={bi['price']:.0f} | "
            f"索引={bi['idx']} | 总笔数={self.stats['total_bi']}"
        )

    def log_pivot(self, pivot: Dict, bi_points: List):
        """记录中枢"""
        if not self.enabled:
            return

        zz = (pivot['zg'] + pivot['zd']) / 2  # 中枢中心
        self._append_csv(self.pivot_file, [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pivot['zg'],
            pivot['zd'],
            zz,
            pivot.get('start', 0),
            pivot['end'],
            pivot.get('status', 'forming')
        ])
        self.stats['total_pivot'] += 1

        # 实时打印中枢
        self.logger.info(
            f"[中枢] ZG={pivot['zg']:.0f} | ZD={pivot['zd']:.0f} | "
            f"ZZ={zz:.0f} | 区间={pivot['zg']-pivot['zd']:.0f}点"
        )

    def log_inclusion(self, before: Dict, after: Dict, direction: int):
        """记录K线包含处理"""
        if not self.enabled:
            return

        dir_str = "向上" if direction == 1 else "向下"
        self.logger.debug(
            f"[包含] {dir_str}处理 | "
            f"前K: H={before['high']:.0f} L={before['low']:.0f} | "
            f"后K: H={after['high']:.0f} L={after['low']:.0f}"
        )
```

### 3.4 信号和交易记录

```python
    def log_signal(self, signal: Dict, reason: str = ""):
        """记录交易信号"""
        if not self.enabled:
            return

        self._append_csv(self.signal_file, [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            signal.get('signal_type', 'Unknown'),
            signal['type'],  # 'Buy' or 'Sell'
            signal['trig'],
            signal['stop'],
            reason
        ])
        self.stats['total_signals'] += 1

        # 醒目打印信号
        direction = "做多" if signal['type'] == 'Buy' else "做空"
        self.logger.warning(
            f"{'='*50}\n"
            f"[信号] {signal.get('signal_type', '')} {direction}\n"
            f"  触发价: {signal['trig']:.0f}\n"
            f"  止损价: {signal['stop']:.0f}\n"
            f"  原因: {reason}\n"
            f"{'='*50}"
        )

    def log_trade(self, action: str, price: float, position: int,
                  pnl: float = 0, signal_type: str = ""):
        """记录交易执行"""
        if not self.enabled:
            return

        self._append_csv(self.trade_file, [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action,  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'
            price,
            position,
            pnl,
            signal_type
        ])

        if 'OPEN' in action:
            self.stats['total_trades'] += 1

        # 醒目打印交易
        self.logger.warning(
            f"[交易] {action} @ {price:.0f} | "
            f"持仓={position} | PnL={pnl:+.0f}"
        )
```

### 3.5 状态监控和摘要

```python
    def log_status(self, status: Dict):
        """记录当前策略状态"""
        if not self.enabled:
            return

        self.logger.info(
            f"[状态] 持仓={status.get('position', 0)} | "
            f"入场价={status.get('entry_price', 0):.0f} | "
            f"止损价={status.get('stop_price', 0):.0f} | "
            f"浮盈={(status.get('unrealized_pnl', 0)):+.0f}"
        )

    def log_chan_state(self, k_lines: List, bi_points: List, pivots: List):
        """记录缠论状态摘要"""
        if not self.enabled:
            return

        self.logger.info(
            f"[缠论] K线={len(k_lines)} | 笔={len(bi_points)} | 中枢={len(pivots)}"
        )

        # 打印最近的笔
        if bi_points:
            recent = bi_points[-3:] if len(bi_points) >= 3 else bi_points
            for i, bi in enumerate(recent):
                bi_type = "顶" if bi['type'] == 'top' else "底"
                self.logger.info(f"  笔{len(bi_points)-len(recent)+i+1}: {bi_type} @ {bi['price']:.0f}")

        # 打印当前中枢
        if pivots:
            pv = pivots[-1]
            self.logger.info(f"  当前中枢: [{pv['zd']:.0f}, {pv['zg']:.0f}]")

    def save_summary(self):
        """保存运行摘要"""
        if not self.enabled:
            return

        self.stats['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = self.debug_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"[摘要] 已保存到 {summary_file}")

    def save_config(self, config: Dict):
        """保存策略配置"""
        if not self.enabled:
            return

        config_file = self.debug_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
```

## 四、策略集成方式

### 4.1 在策略初始化时创建Debugger

```python
class CtaChanPivotStrategy(CtaTemplate):

    # 添加debug参数
    debug_enabled = True

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 初始化debugger
        if self.debug_enabled:
            self.debugger = ChanDebugger(
                strategy_name=strategy_name,
                base_dir="data/debug",
                enabled=True,
                log_level="DEBUG"
            )
            # 保存配置
            self.debugger.save_config({
                "strategy_name": strategy_name,
                "vt_symbol": vt_symbol,
                "settings": setting
            })
        else:
            self.debugger = None
```

### 4.2 在K线回调中记录数据

```python
    def on_bar(self, bar: BarData):
        # 记录1分钟K线
        if self.debugger:
            self.debugger.log_kline_1m({
                'datetime': bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                'open': bar.open_price,
                'high': bar.high_price,
                'low': bar.low_price,
                'close': bar.close_price,
                'volume': bar.volume
            })

        # ... 策略逻辑 ...

        # 记录5分钟K线
        if self.debugger and is_5m_bar:
            self.debugger.log_kline_5m(bar_5m_dict)
```

### 4.3 在信号生成时记录

```python
    def _check_signal(self):
        # ... 信号检测逻辑 ...

        if signal_generated and self.debugger:
            self.debugger.log_signal(
                signal={'type': 'Buy', 'trig': trigger_price, 'stop': stop_price, 'signal_type': '3B'},
                reason=f"3类买点: 回踩不破中枢高点{pivot_zg}"
            )
```

### 4.4 在策略停止时保存摘要

```python
    def on_stop(self):
        if self.debugger:
            self.debugger.save_summary()
        super().on_stop()
```

## 五、日志输出示例

```
09:30:05 | [5M] 2025-01-24 09:30:00 | C=1245 | MACD=0.85 (金叉) | ATR=8.5
09:30:05 | [缠论] K线=156 | 笔=12 | 中枢=3
09:30:05 |   笔10: 底 @ 1238
09:30:05 |   笔11: 顶 @ 1252
09:30:05 |   笔12: 底 @ 1243
09:30:05 |   当前中枢: [1240, 1250]
09:35:00 | [笔] 新顶分型 | 价格=1256 | 索引=158 | 总笔数=13
09:35:00 | [中枢] ZG=1250 | ZD=1240 | ZZ=1245 | 区间=10点
09:40:00 | ==================================================
09:40:00 | [信号] 3B 做多
09:40:00 |   触发价: 1248
09:40:00 |   止损价: 1238
09:40:00 |   原因: 3类买点: 回踩不破中枢高点1250
09:40:00 | ==================================================
09:41:00 | [交易] OPEN_LONG @ 1248 | 持仓=1 | PnL=+0
09:45:00 | [状态] 持仓=1 | 入场价=1248 | 止损价=1238 | 浮盈=+5
```

## 六、数据分析支持

### 6.1 回放分析脚本

```python
# scripts/analyze_debug.py
import pandas as pd
from pathlib import Path

def analyze_debug_session(debug_dir: str):
    """分析debug会话数据"""
    debug_path = Path(debug_dir)

    # 加载数据
    kline_5m = pd.read_csv(debug_path / "kline_5m.csv")
    bi_data = pd.read_csv(debug_path / "chan_bi.csv")
    pivot_data = pd.read_csv(debug_path / "chan_pivot.csv")
    signals = pd.read_csv(debug_path / "signals.csv")
    trades = pd.read_csv(debug_path / "trades.csv")

    print(f"5分钟K线: {len(kline_5m)} 条")
    print(f"笔: {len(bi_data)} 个")
    print(f"中枢: {len(pivot_data)} 个")
    print(f"信号: {len(signals)} 个")
    print(f"交易: {len(trades)} 笔")

    # 信号分析
    if not signals.empty:
        print("\n信号分布:")
        print(signals.groupby('signal_type').size())

    # 盈亏分析
    if not trades.empty:
        total_pnl = trades['pnl'].sum()
        print(f"\n总盈亏: {total_pnl:.0f}")
```

## 七、配置参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|-----|
| debug_enabled | bool | True | 是否启用debug |
| debug_dir | str | "data/debug" | debug目录 |
| log_level | str | "DEBUG" | 日志级别 |
| log_console | bool | True | 是否打印到控制台 |
| save_kline | bool | True | 是否保存K线 |
| save_bi | bool | True | 是否保存笔 |
| save_pivot | bool | True | 是否保存中枢 |

## 八、注意事项

1. **磁盘空间**: 长时间运行会产生大量日志，建议定期清理或设置日志轮转
2. **性能影响**: Debug模式会有少量性能开销，生产环境可关闭
3. **数据安全**: debug目录可能包含敏感交易数据，注意权限控制
4. **时区处理**: 确保datetime使用统一时区(建议使用交易所时区)

## 九、扩展功能

### 9.1 可视化支持
后续可增加实时图表展示:
- K线图 + 笔 + 中枢叠加
- MACD指标图
- 资金曲线图

### 9.2 告警功能
- 重要信号微信/钉钉推送
- 异常状态邮件告警

### 9.3 远程监控
- WebSocket实时推送
- 移动端监控APP
