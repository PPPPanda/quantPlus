# Development Guide

本文档提供 quantPlus 项目的开发规范和技术指南。

---

## 1. 项目定位与原则

**本仓库是 quantPlus 主工程**，面向 A 股期货量化交易（主要品种：棕榈油，大商所 DCE）。

### 核心原则

- **vendor/vnpy 只做 submodule + editable install**：原则上不直接修改 vendor/vnpy 源码；如需定制，通过继承/包装在 src/qp 中实现。
- **业务代码只允许放在 src/qp 目录**：禁止在 vendor/vnpy 下开发任何业务逻辑。
- **双网关支持**：CTP (SimNow) 和 TTS (OpenCTP) 灵活切换，策略代码完全兼容。
- **研究与执行解耦**：
  - OpenBB 用于研究层（因子计算、数据抓取、探索性分析）
  - 实盘/回测数据以 vn.py 数据库/CSV 导入为准
  - 两者通过 ingest 脚本桥接，不在运行时混用

---

## 2. 目录结构

```
quantPlus/
├── vendor/
│   └── vnpy/                              # vn.py submodule，独立维护与更新
├── src/
│   └── qp/
│       ├── runtime/
│       │   └── trader_app.py              # 实盘启动 + GUI 显示入口（支持 profile 和 gateway 参数）
│       ├── ui/
│       │   ├── launcher.py                # GUI 启动器
│       │   └── profiles.py                # Profile 配置（trade/research/all）
│       ├── strategies/
│       │   ├── base.py                    # 策略基类/通用工具
│       │   ├── cta_palm_oil.py            # 双均线策略（学习用）
│       │   ├── cta_turtle_enhanced.py     # 增强海龟策略
│       │   └── cta_chan_pivot.py          # 缠论中枢策略（主力策略）
│       ├── datafeed/
│       │   ├── normalizer.py              # 1分钟K线归一化（多源一致性）
│       │   ├── bar_generator.py           # PandasStyleBarGenerator（K线合成）
│       │   ├── session_synthesizer.py     # 交易时段K线合成器
│       │   ├── xtquant_feed.py            # 迅投研数据源封装
│       │   ├── download_palm_oil.py       # 棕榈油数据下载
│       │   └── base.py                    # 数据源基类
│       ├── research/
│       │   ├── openbb_fetch.py            # OpenBB 数据拉取 / 因子计算
│       │   ├── ingest_vnpy.py             # 数据 -> vn.py 数据库/CSV 转换入库
│       │   └── pipeline_palm_oil.py       # 一键数据流水线
│       ├── backtest/
│       │   ├── cli.py                     # CLI 回测入口（python -m qp.backtest.cli）
│       │   ├── engine.py                  # 回测引擎封装
│       │   ├── run_cta_backtest.py        # 脚本化回测（批处理/CI）
│       │   ├── run_tick_backtest.py        # Tick 级别回测
│       │   └── run_xtquant_backtest.py    # 迅投研数据回测
│       ├── apps/
│       │   └── enhanced_chart/            # 增强K线图 App（MACD/MA/OI叠加）
│       ├── common/
│       │   ├── constants.py               # 全局常量
│       │   ├── logging.py                 # 日志配置
│       │   └── utils.py                   # 通用工具函数
│       └── utils/
│           └── chan_debugger.py            # 缠论调试工具（K线/笔/中枢/信号记录）
├── strategies/                            # 桥接文件（vnpy_ctastrategy 加载目录）
│   ├── cta_palm_oil.py                    # → qp.strategies.cta_palm_oil
│   ├── cta_turtle_enhanced.py             # → qp.strategies.cta_turtle_enhanced
│   └── cta_chan_pivot.py                  # → qp.strategies.cta_chan_pivot
├── scripts/
│   ├── check_sensitive_info.py            # pre-commit 敏感信息检测
│   ├── convert_png_to_ico.py              # PNG → ICO 转换工具
│   ├── run_chan_pivot.py                  # 缠论策略基准运行脚本
│   ├── test_chan_debugger.py              # ChanDebugger 功能测试
│   ├── test_chan_invariants.py            # 缠论结构不变量测试
│   └── archive/                           # 已归档的一次性实验脚本（29个）
├── tests/
│   └── test_openctp_connection.py         # OpenCTP 连接测试脚本
├── data/
│   ├── analyse/                           # 分析用CSV数据
│   │   ├── wind/                          # Wind数据源CSV
│   │   └── *.csv                          # XTQuant数据源CSV
│   └── debug/                             # ChanDebugger 输出目录
├── experiments/                           # GUI/回测实验截图与产物
├── docs/
│   ├── development.md                     # 本文件
│   ├── debug.md                           # 调试指南
│   ├── enhanced_chart_development.md      # 增强K线图开发文档
│   ├── ctptest_guide.md                   # CTP测试指南
│   ├── openctp_quickstart.md              # OpenCTP 快速上手指南
│   ├── openctp_integration_research.md    # OpenCTP 技术调研报告
│   ├── trade_all_features.md              # Trade 模式功能说明
│   ├── quickstart_chart_recorder.md       # K线图和数据录制快速上手
│   └── data_directory_guide.md            # Data 目录说明
├── pyproject.toml                         # uv 依赖管理
└── uv.lock                                # 锁定依赖版本
```

### 职责说明

| 路径 | 职责 |
|------|------|
| `vendor/vnpy/` | vn.py 框架 submodule，只读引用，更新通过 git submodule update |
| `src/qp/runtime/trader_app.py` | GUI 主入口，支持 `--profile` (trade/research/all) 和 `--gateway` (ctp/tts) 参数 |
| `src/qp/strategies/cta_chan_pivot.py` | **缠论中枢策略**，基于5m笔/中枢信号+15m MACD过滤+ATR风控（主力策略） |
| `src/qp/strategies/cta_palm_oil.py` | 双均线 CTA 策略，继承 CtaTemplate（学习用） |
| `src/qp/strategies/cta_turtle_enhanced.py` | 增强海龟策略（趋势过滤+中轨止盈） |
| `src/qp/datafeed/normalizer.py` | **1分钟K线归一化**：时间戳标准化、交易时段过滤、session-aware窗口计算 |
| `src/qp/datafeed/bar_generator.py` | PandasStyleBarGenerator，pandas风格K线合成（label=right） |
| `src/qp/datafeed/session_synthesizer.py` | 交易时段K线合成器（1m→session bar→daily bar） |
| `src/qp/utils/chan_debugger.py` | 缠论调试工具，记录K线/笔/中枢/信号/交易到 `data/debug/` |
| `src/qp/backtest/cli.py` | CLI 回测入口（`python -m qp.backtest.cli`） |
| `src/qp/backtest/engine.py` | 回测引擎封装（run_backtest函数） |
| `src/qp/research/openbb_fetch.py` | 研究层数据获取，与交易运行时隔离 |
| `src/qp/research/ingest_vnpy.py` | 数据桥接：将数据转为 vn.py 可用格式 |
| `scripts/archive/` | 已归档的一次性实验脚本（iteration_v1-v6、backtest_v2-v4、trim等） |

---

## 3. 环境要求

- **Python 版本**：优先 3.13，但 3.11+ 可用
- **操作系统**：不限定；GUI 功能需要桌面环境（Qt/PySide6）
- **依赖管理**：**uv 是唯一入口**，不要混用 pip/conda 直接安装依赖
  - 如使用 conda，仅作为底层 Python 解释器来源，依赖仍由 uv 锁定管理

---

## 4. uv 常用命令

### 依赖同步

```bash
# 基础同步（不含 GUI）
uv sync

# 完整同步（含 GUI 与交易组件）
uv sync --all-extras
```

### 运行 Trader GUI

```bash
# 全功能模式（实盘 + 回测 + 数据管理），默认 CTP 网关
uv run python -m qp.runtime.trader_app --profile all

# 仅投研/回测（加载 CtaBacktesterApp, DataManagerApp）
uv run python -m qp.runtime.trader_app --profile research

# 仅实盘交易（加载 CtaStrategyApp, RiskManagerApp）
uv run python -m qp.runtime.trader_app --profile trade

# 使用 OpenCTP TTS 网关（7x24 模拟环境）
uv run python -m qp.runtime.trader_app --gateway tts

# 组合使用：TTS 网关 + 投研模式
uv run python -m qp.runtime.trader_app --gateway tts --profile research
```

### 脚本化回测

```bash
# 批量回测入口（将来采用该入口）
uv run python -m qp.backtest.run_cta_backtest
```

---

## 5. vn.py GUI 使用说明

### App 与页签

GUI 的功能页签由加载的 App 模块决定：

| App 类 | 页签功能 |
|--------|----------|
| `CtaStrategyApp` | CTA 策略实盘运行 |
| `CtaBacktesterApp` | CTA 策略回测 |
| `DataManagerApp` | 历史数据管理（查看/导入/导出） |
| `RiskManagerApp` | 风控规则管理 |
| `ChartWizardApp` | K线图表 |

### Profile 含义

| Profile | 用途 | 加载的 App |
|---------|------|-----------|
| `trade` | 实盘交易 | CtaStrategy, RiskManager, ChartWizard |
| `research` | 投研回测 | CtaBacktester, DataManager, ChartWizard |
| `all` | 全功能调试 | 以上全部 |

### vt_symbol 规范

**格式**：`合约代码.交易所`

**示例**：`p2405.DCE`（棕榈油 2405 合约，大商所）

- `DCE` = 大连商品交易所（大商所）
- `p` = 棕榈油品种代码

**重要**：策略、回测、数据入库、GUI 各处必须使用完全一致的 vt_symbol。

### 常见问题排查

| 症状 | 可能原因 | 解决方法 |
|------|----------|----------|
| 策略找不到合约 | vt_symbol 格式错误或合约未订阅 | 登录后在"合约查询"中确认 `pXXXX.DCE` 存在 |
| 回测加载不到数据 | 数据库中 vt_symbol 与策略不一致 | 检查 DataManager 中的数据 symbol 格式 |
| K线图表空白 | 未订阅行情或历史数据缺失 | 先通过 DataManager 确认数据已入库 |

---

## 6. 编码规范

### Python 风格

- **类型标注**：所有函数签名必须有完整的 typing 标注
- **数据结构**：优先使用 `dataclass` 或 `pydantic.BaseModel`
- **模块边界**：严格区分 runtime / strategies / research / backtest，禁止跨层直接耦合

### 日志

- **统一使用 `logging` 模块**，禁止 `print` 作为主日志
- 关键路径必须有 `info` / `warning` / `error` 级别日志
- 日志格式应包含时间戳、模块名、级别

### 错误处理

- **外部依赖**（OpenBB、网络请求、数据源）：必须显式捕获异常，提供降级或明确报错
- **交易逻辑**：严禁 silent failure，任何异常必须记录并通知

### 依赖管理

- 禁止依赖未锁定的隐式系统包
- 所有依赖必须通过 pyproject.toml 声明，由 uv 管理

### 敏感信息保护

**原则**：**禁止提交任何真实的账号密码、Token、私钥等敏感信息到 Git 仓库**

**自动检测**：
- 项目已集成 `scripts/check_sensitive_info.py` 敏感信息检测脚本
- 在 `git commit` 时通过 pre-commit hook 自动运行
- 检测内容：
  - 真实密码（排除 `your_password`、`123456` 等占位符）
  - Token/API Key（长度 ≥ 32 字符的随机字符串）
  - 私钥（PEM 格式）
  - 数据库连接字符串（包含真实密码）
  - 不应提交的文件类型（`.vntrader/*.json`、`*.pem`、`*.key` 等）

**配置文件保护**：
- `.vntrader/connect_ctp.json`（SimNow 配置）
- `.vntrader/connect_tts.json`（OpenCTP 配置）
- 所有 `.vntrader/*.json` 文件已在 `.gitignore` 中

**文档中的示例**：
- 必须使用占位符：`your_username`、`your_password`、`your_token`
- **禁止**在文档中写入任何真实账号密码（包括测试账号）
- 用户需自行申请：[SimNow 官网](https://www.simnow.com.cn) / [OpenCTP GitHub](https://github.com/krenx1983/openctp)

**手动测试**：
```bash
# 测试敏感信息检测
python scripts/check_sensitive_info.py

# 强制提交（不推荐，仅用于紧急情况）
git commit --no-verify
```

**如何修复检测失败**：
1. 移除暂存区中包含敏感信息的文件：`git restore --staged <file>`
2. 使用占位符替换敏感信息
3. 确保敏感配置文件已在 `.gitignore` 中

---

## 7. Smoke Tests

### 立即可跑

```bash
# 验证 vnpy 导入
uv run python -c "import vnpy; print(vnpy.__version__ if hasattr(vnpy, '__version__') else 'vnpy-ok')"

# 验证 OpenBB 导入
uv run python -c "from openbb import obb; print('openbb-ok')"

# 验证 PySide6/Qt 可用
uv run python -c "from PySide6.QtWidgets import QApplication; print('pyside6-ok')"

# 测试 OpenCTP 连接（需要配置 .vntrader/connect_tts.json）
uv run python tests/test_openctp_connection.py
```

### 待 src/qp 模块就绪后可跑

```bash
# Trader GUI 帮助信息
uv run python -m qp.runtime.trader_app --help

# 策略模块导入验证
uv run python -c "from qp.strategies.cta_palm_oil import CtaPalmOilStrategy; print('strategy-ok')"
uv run python -c "from qp.strategies.cta_turtle_enhanced import CtaTurtleEnhancedStrategy; print('turtle-ok')"

# 回测脚本帮助
uv run python -m qp.backtest.run_cta_backtest --help

# 60分钟数据回测验证
uv run python -m qp.backtest.run_cta_backtest --strategy CtaTurtleEnhancedStrategy --interval HOUR --days 240
```

---

## 8. 策略开发规范

### 策略文件结构

所有 CTA 策略放置于 `src/qp/strategies/` 目录，命名规范：`cta_{品种}_{策略类型}.py`

### 策略模板要点

```python
from typing import Optional
from vnpy.trader.object import BarData, TickData
from vnpy.trader.utility import ArrayManager, BarGenerator
from vnpy.trader.constant import Interval
from vnpy_ctastrategy import CtaTemplate

class MyStrategy(CtaTemplate):
    author = "QuantPlus"

    # 参数（GUI 可见）
    param1: int = 10
    bar_window: int = 1           # K线窗口（1=1分钟, 15=15分钟）
    bar_interval: str = "MINUTE"  # K线周期类型
    parameters = ["param1", "bar_window", "bar_interval"]

    # 变量（GUI 可见）
    var1: float = 0.0
    variables = ["var1"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        # ArrayManager size 必须覆盖所有窗口 + buffer
        self.am = ArrayManager(size=self.param1 + 20)
        # K线生成器（实盘使用）
        self.bg: Optional[BarGenerator] = None

    def on_init(self):
        # 创建 K 线生成器（实盘时 Tick -> Bar）
        interval = Interval.HOUR if self.bar_interval == "HOUR" else Interval.MINUTE
        if self.bar_window <= 1 and interval == Interval.MINUTE:
            self.bg = BarGenerator(self.on_bar)
        else:
            self.bg = BarGenerator(
                self.on_bar,
                window=self.bar_window,
                on_window_bar=self.on_bar,
                interval=interval,
            )
        self.load_bar(10)  # 预加载历史数据

    def on_tick(self, tick: TickData):
        """实盘时由 CTA 引擎调用，将 Tick 转换为 Bar"""
        if self.bg:
            self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return  # warm-up 未完成，不交易
        # 交易逻辑...
```

### K 线生成机制（重要）

**回测与实盘的数据流差异**：

| 场景 | 数据流 | 说明 |
|------|--------|------|
| 回测 | 数据库 → BacktestingEngine → `on_bar()` | 引擎直接推送历史 Bar 数据 |
| 实盘 | 交易所 → CTA Engine → `on_tick()` → BarGenerator → `on_bar()` | 需要手动转换 Tick → Bar |

**关键点**：
- 回测时 `on_bar()` 由引擎直接调用，`on_tick()` 不会被调用
- 实盘时 CTA 引擎只推送 `on_tick()`，**不会自动调用 `on_bar()`**
- 因此策略**必须**实现 `BarGenerator` 来支持实盘交易

**BarGenerator 工作原理**：
```
Tick 数据 → BarGenerator.update_tick() → 1 分钟 Bar → on_bar()
                                       ↓
                                   （可选）合成 N 分钟/小时 Bar → on_window_bar()
```

**K 线周期配置**：
- `bar_window=1, bar_interval="MINUTE"`: 1 分钟线（默认）
- `bar_window=15, bar_interval="MINUTE"`: 15 分钟线
- `bar_window=1, bar_interval="HOUR"`: 1 小时线

### K 线合成时间戳对齐（重要）

**问题背景**：

VNPY 原生 `BarGenerator` 与 pandas `resample` 的 K 线时间戳对齐方式不同，导致策略在不同环境下结果不一致。

| 方式 | 时间戳风格 | 09:01-09:05 数据的时间戳 | 说明 |
|------|-----------|------------------------|------|
| VNPY BarGenerator | `label='left'` | **09:00** | 窗口开始时间 |
| pandas resample | `label='right'` | **09:05** | 窗口结束时间 |

**影响**：时间戳差异会导致分型检测、MACD 对齐等逻辑在相同数据上产生不同结果。

**解决方案：PandasStyleBarGenerator**

为保证回测与研究代码（pandas）结果一致，项目提供 `PandasStyleBarGenerator`：

```python
class PandasStyleBarGenerator:
    """
    pandas 风格的 K 线合成器.

    与 pandas resample('5min', label='right', closed='right') 一致：
    - 09:01-09:05 的数据合成为时间戳 09:05 的 K 线
    """

    def __init__(self, window: int, on_window_bar, on_bar=None):
        self.window = window
        self.on_window_bar = on_window_bar
        # ...

    def _get_window_end(self, dt: datetime) -> datetime:
        """计算窗口结束时间（用作 K 线时间戳）"""
        minute = dt.minute
        window_start_minute = (minute // self.window) * self.window
        window_end = dt.replace(minute=window_start_minute, second=0, microsecond=0)
        window_end += timedelta(minutes=self.window)
        return window_end

    def update_bar(self, bar: BarData) -> None:
        """从 1 分钟 K 线合成 N 分钟 K 线（回测用）"""
        window_end = self._get_window_end(bar.datetime)
        # 使用窗口结束时间作为时间戳
        self._current_bar = BarData(datetime=window_end, ...)

    def update_tick(self, tick: TickData) -> None:
        """从 Tick 合成 K 线（实盘用）"""
        # Tick → 1m Bar → Nm Bar
```

**使用场景**：

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 与 pandas 研究代码对比验证 | `PandasStyleBarGenerator` | 时间戳一致，结果可复现 |
| 纯 VNPY 环境 | 原生 `BarGenerator` | 兼容性好 |
| 多周期策略（如 5m + 15m MACD） | `PandasStyleBarGenerator` | 避免周期对齐偏差 |

**注意事项**：

1. **回测与实盘一致性**：`PandasStyleBarGenerator` 同时实现了 `update_bar()`（回测）和 `update_tick()`（实盘），确保两者 K 线合成逻辑一致
2. **输入必须是 1 分钟 K 线**：策略的 `on_bar()` 接收 1 分钟数据，由合成器生成 N 分钟 K 线
3. **VNPY BarGenerator 已知问题**：小时线合成存在边界 bug（GitHub Issue #2775），建议使用自定义合成器

### 工程最佳实践

| 要点 | 说明 |
|------|------|
| **BarGenerator 必需** | 实盘交易必须在 `on_tick()` 中调用 `bg.update_tick()` |
| **Warm-up 防护** | `ArrayManager size = max(所有窗口) + 20`，防止 0 成交 |
| **Look-ahead 防护** | 计算指标时使用 `[-window-1:-1]` 切片，避免使用当前 bar |
| **止损单管理** | 每根 bar 调用 `cancel_all()` 后重新挂止损单 |
| **日志记录** | 入场/出场/加仓必须 `write_log()`，便于调试 |
| **参数校验** | `on_init()` 中校验参数合法性 |

### GUI 可见性

策略需要放置桥接文件才能在 GUI 中显示。

**重要**：vnpy_ctastrategy 从 `Path.cwd().joinpath("strategies")` 加载策略，即**项目根目录下的 `strategies/` 文件夹**，而不是 `.vntrader/strategies/`。

```bash
# strategies/{策略文件名}.py （项目根目录）
from qp.strategies.{策略模块} import {策略类名}
```

**示例**：
```
quantPlus/
├── strategies/                    # vnpy_ctastrategy 加载策略的目录
│   ├── cta_chan_pivot.py         # 桥接文件
│   ├── cta_palm_oil.py          # 桥接文件
│   └── cta_turtle_enhanced.py   # 桥接文件
└── src/qp/strategies/            # 策略源码目录
    ├── cta_chan_pivot.py         # 缠论中枢策略
    ├── cta_palm_oil.py          # 双均线策略
    └── cta_turtle_enhanced.py   # 增强海龟策略
```

**桥接文件内容**（只需一行导入）：
```python
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
```

**注意**：修改桥接文件后，需要重启 GUI 才能生效。如果仍不显示，尝试删除 `strategies/__pycache__/` 缓存目录。

**关于 `.vntrader/` 目录**：

`.vntrader/` 目录是 vnpy 的**配置文件目录**，**不是策略加载目录**。它用于存储：
- 网关连接配置：`connect_ctp.json`、`connect_tts.json`
- 策略运行配置：`cta_strategy_setting.json`
- 策略运行数据：`cta_strategy_data.json`
- 全局设置：`vt_setting.json`

**实盘和回测都从同一个 `strategies/` 目录加载策略**，没有区分。

### 实盘运行检查清单

**启动前确认**：
- [ ] 网关连接成功（日志: "行情服务器登录成功" + "交易服务器登录成功"）
- [ ] 合约信息已加载（日志: "行情订阅" 无失败警告）
- [ ] 策略初始化完成（日志: "策略初始化完成"）
- [ ] ArrayManager 已初始化（通常需要 max_window + buffer_size 根 K 线）
- [ ] 策略已启动（`strategy.trading = True`）
- [ ] Tick 数据正常推送（日志: on_tick 调用）
- [ ] K 线成功生成（日志: on_bar 调用）

**常见问题排查**：

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 策略不交易 | `trading=False` | 确认已点击"启动"按钮 |
| 没有收到行情 | 合约未订阅/网关断开 | 检查日志中的行情订阅记录 |
| 历史数据加载失败 | 数据源不可用 | 在 `load_bar()` 中设置 `use_database=True` |
| K 线不生成 | BarGenerator 配置错误 | 检查 `bar_window` 和 `bar_interval` 参数 |
| ArrayManager 不初始化 | 历史数据不足 | 增加 `load_bar()` 的天数 |

### 回测验证

新策略必须通过脚本化回测验证：

```bash
uv run python -m qp.backtest.run_cta_backtest --strategy {策略类名} --days 365 -v
```

---

## 9. 现有策略说明

### 策略回测对比

| 策略 | 数据周期 | 回测区间 | Sharpe | 总收益率 | 年化收益 | 最大回撤 | 定位 |
|------|---------|---------|--------|---------|---------|---------|------|
| CtaChanPivotStrategy (p2601) | 1m→5m/15m | 2025-07~2026-01 | **2.74** | **+11.53%** | **+31.44%** | -3.23% | **主力策略** |
| CtaTurtleEnhancedStrategy (激进) | 60m | 2025-05~2026-01 | 1.28 | +24.41% | +34.07% | -8.23% | 趋势跟踪 |
| CtaPalmOilStrategy | 日线 | 2025-01~2026-01 | -1.50 | -1.64% | -2.28% | -1.85% | 学习用 |

> **注**：ChanPivot 回测数据来自 13 合约批量回测中表现最佳的 p2601.DCE（XT 数据源）。13 合约总体：4 盈利 / 13 总计，策略仍在迭代优化中。

### CtaTurtleEnhancedStrategy（增强海龟策略）**推荐**

改进版海龟策略，经激进优化后收益率达24%+，年化34%：

- **趋势过滤**：MA10 > MA100 时只做多，MA10 < MA100 时只做空（超长周期过滤噪音）
- **中轨止盈**：触及唐奇安通道中线时止盈，锁定利润
- **极速出场**：exit_window=1，最快响应反转信号
- **激进仓位**：risk_per_trade=6%，max_units=50
- **快速加仓**：pyramid_atr=0.15，利润快速累积

**默认参数**（2026-01 激进优化: Sharpe 1.28, 收益 +24.41%）：

| 参数 | 值 | 说明 |
|------|-----|------|
| entry_window | 15 | 入场通道窗口 |
| exit_window | 1 | 出场通道窗口（极速出场） |
| trend_ma_fast | 10 | 快速趋势均线 |
| trend_ma_slow | 100 | 慢速趋势均线（超长周期过滤） |
| atr_stop | 1.5 | ATR 止损倍数 |
| risk_per_trade | 0.06 | 单笔风险预算 6% |
| max_units | 50 | 最大持仓手数 |
| pyramid_atr | 0.15 | 加仓间隔 ATR 倍数 |
| use_trend_filter | True | 趋势过滤开关 |
| use_mid_line_exit | True | 中轨止盈开关 |
| enable_pyramid | True | 金字塔加仓开关 |

**保守版参数**（Sharpe 1.14, 收益 +6.85%, 回撤 -2.92%）：
- exit_window: 2
- trend_ma_slow: 80
- risk_per_trade: 0.03
- max_units: 15
- pyramid_atr: 0.5

**回测命令**：
```bash
uv run python -m qp.backtest.run_cta_backtest --strategy CtaTurtleEnhancedStrategy --interval HOUR --days 240
```

### CtaPalmOilStrategy

基础双均线策略，用于学习和验证框架功能。

### CtaChanPivotStrategy（缠论中枢策略）

#### 策略概述

基于缠论中枢理论的 CTA 策略，核心流程：

```
5分钟K线 → 包含处理 → 分型识别 → 严格笔构建 → 中枢(ZhongShu)识别
  → 3B/3S 主信号 + 2B/2S 辅助信号
  → 15分钟 MACD 趋势过滤
  → P1硬止损 + ATR移动止损
```

策略实现完全增量化：所有指标（MACD、ATR）和缠论结构（包含处理、分型、笔、中枢）均为逐 bar 增量更新，无需回溯历史数据。

#### 数据流架构

```
┌─────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│ 1分钟K线输入 │───→│ session-aware 合成     │───→│ 增量指标计算        │
│  (on_bar)   │    │  5m: _update_5m_bar() │    │  MACD(5m/15m)     │
│             │    │ 15m: _update_15m_bar()│    │  ATR(5m)          │
└─────────────┘    └──────────────────────┘    └───────────────────┘
                                                        │
      ┌─────────────────────────────────────────────────┘
      ▼
┌───────────────────┐    ┌───────────────────┐    ┌──────────────┐
│ 缠论结构增量构建    │───→│ 信号检测            │───→│ 风控与执行     │
│  包含处理          │    │  3B/3S (中枢信号)   │    │  P1硬止损     │
│  分型 → 严格笔     │    │  2B/2S (背驰辅助)   │    │  ATR移动止损  │
│  中枢识别          │    │  15m MACD过滤       │    │  1m级别监控   │
└───────────────────┘    └───────────────────┘    └──────────────┘
```

**关键设计**：
- 1 分钟 K 线通过 `normalizer.py` 的 session-aware 逻辑合成 5m/15m，保证跨数据源（Wind / XTQuant）一致
- 15m MACD 使用 `shift(1)` 延迟（`_prev_diff_15m` / `_prev_dea_15m`），避免 look-ahead bias
- 信号在 5m bar 级别生成，入场和止损在 1m bar 级别监控执行

#### 核心信号说明

**3B 买点（中枢三买）**：

| 条件 | 说明 |
|------|------|
| 回踩点 > ZG | 当前笔端点（底分型）高于中枢高点，说明回踩不破中枢 |
| 离开段 > ZG | 前一笔端点（顶分型）也高于中枢高点，说明确实向上离开过 |
| 中枢时效性 | `end_bi_idx >= len(bi_points) - pivot_valid_range`，中枢必须是近期的 |
| 大周期多头 | 15分钟 MACD：`prev_diff_15m > prev_dea_15m`（金叉状态） |

**3S 卖点（中枢三卖）**：

| 条件 | 说明 |
|------|------|
| 回抽点 < ZD | 当前笔端点（顶分型）低于中枢低点，说明回抽不破中枢 |
| 离开段 < ZD | 前一笔端点（底分型）也低于中枢低点，说明确实向下离开过 |
| 中枢时效性 | 同上 |
| 大周期空头 | 15分钟 MACD：`prev_diff_15m < prev_dea_15m`（死叉状态） |

**2B/2S 辅助信号（趋势延续）**：

| 信号 | 条件 |
|------|------|
| 2B 买点 | 低点抬高（`p_now.price > p_prev.price`）+ MACD 背驰（`diff` 回升）+ 大周期多头 |
| 2S 卖点 | 高点降低（`p_now.price < p_prev.price`）+ MACD 背驰（`diff` 回落）+ 大周期空头 |

> 2B/2S 仅在 3B/3S 未触发时作为补充信号，优先级低于中枢信号。

#### 风控系统

**P1 硬止损**：

- 多头止损 = 回踩点价格 - 1（`stop_base - 1`）
- 空头止损 = 回抽点价格 + 1（`stop_base + 1`）
- 在 1 分钟级别逐 bar 检查（`_check_stop_loss_1m`）

**ATR 移动止损**：

| 阶段 | 条件 | 行为 |
|------|------|------|
| 未激活 | 浮盈 ≤ `atr_activate_mult × ATR`（默认 1.5 倍） | 仅使用 P1 硬止损 |
| 激活后 | 浮盈 > 1.5 × ATR | 启动追踪，多头：`high - atr_trailing_mult × ATR`；空头：`low + atr_trailing_mult × ATR` |
| 追踪中 | 每根 5m bar | 止损只能朝有利方向移动（多头只上移，空头只下移） |

**入场过滤**：触发价与止损距离不超过 `atr_entry_filter × ATR`（默认 2.0 倍），过滤距离过大的信号。

#### 可配置参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `macd_fast` | 12 | MACD 快线 EMA 周期 |
| `macd_slow` | 26 | MACD 慢线 EMA 周期 |
| `macd_signal` | 9 | MACD 信号线 EMA 周期 |
| `atr_window` | 14 | ATR 计算窗口（5m bar 数） |
| `atr_trailing_mult` | 3.0 | ATR 移动止损倍数 |
| `atr_activate_mult` | 1.5 | 激活移动止损的浮盈 ATR 倍数 |
| `atr_entry_filter` | 2.0 | 入场过滤：触发价与止损距离上限（ATR 倍数） |
| `min_bi_gap` | 4 | 严格笔端点最小间隔（包含处理后的 K 线数） |
| `pivot_valid_range` | 6 | 中枢有效范围（笔端点数，超过则视为过期） |
| `fixed_volume` | 1 | 固定开仓手数 |
| `debug` | False | 是否输出 debug 日志到控制台 |
| `debug_enabled` | True | 是否启用 ChanDebugger 记录 |
| `debug_log_console` | True | ChanDebugger 是否同时输出到控制台 |

#### 回测表现

批量回测（13 合约，1 分钟数据）：

| 合约 | 收益率 | Sharpe | 最大回撤 | 备注 |
|------|--------|--------|----------|------|
| **p2601.DCE** | **+11.53%** | **2.74** | **-3.23%** | 最佳合约 |
| 其他 12 合约 | — | — | — | 混合表现 |
| **总体** | — | — | — | **4 盈利 / 13 总计** |

> 策略对趋势行情敏感度高，在震荡市中表现一般。最佳表现出现在趋势明显的合约上。

#### 使用方法

**CLI 回测**：

```bash
# 单合约回测
uv run python -m qp.backtest.run_cta_backtest \
  --strategy CtaChanPivotStrategy \
  --symbol p2601 \
  --interval MINUTE \
  --days 365

# 批量回测（多合约）
uv run python -m qp.backtest.run_cta_backtest \
  --strategy CtaChanPivotStrategy \
  --interval MINUTE \
  --days 365 \
  --batch
```

**桥接文件**（`strategies/cta_chan_pivot.py`）：

```python
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
```

#### Debug 系统

策略内置 `ChanDebugger` 调试工具，输出目录为 `data/debug/`，包含以下记录：

| 文件类型 | 内容 | 用途 |
|----------|------|------|
| K线记录 | 1m / 5m K 线及指标值 | 验证数据合成正确性 |
| 包含处理 | 合并前后的 high/low 和方向 | 调试包含处理逻辑 |
| 笔端点 | 类型(top/bottom)、价格、索引 | 验证严格笔构建 |
| 中枢 | ZG/ZD、起止笔索引、状态 | 验证中枢识别 |
| 信号 | 类型(3B/3S/2B/2S)、方向、触发价、止损价 | 分析信号质量 |
| 交易记录 | 开平仓、价格、手数、盈亏、信号类型 | 复盘交易表现 |
| 持仓状态 | 方向、入场价、止损价、浮盈、追踪状态 | 实时监控 |
| 策略配置 | 所有参数快照 | 回测复现 |

**启用方式**：设置 `debug_enabled=True`（默认已启用），输出到 `data/debug/{strategy_name}/` 目录。

## 10. 数据获取

### 数据源

| 数据源 | 用途 | 说明 |
|--------|------|------|
| 迅投研 (XTQuant) | **主数据源** | Tick/分钟数据，支持 Token 连接 |
| akshare (新浪) | 备用 | 有限历史深度，无 Tick |

### 迅投研 Token 模式配置

迅投研支持两种连接方式：
1. **QMT 客户端模式**：需要启动 QMT 客户端
2. **Token 模式**：无需客户端，直接使用 Token 连接（推荐）

**Token 配置文件**：`.vntrader/vt_setting.json`

```json
{
    "datafeed.name": "xt",
    "datafeed.username": "token",
    "datafeed.password": "<your_xtquant_token>"
}
```

**Token 获取**：登录迅投研官网或 QMT 客户端获取 API Token。

### 迅投研数据下载方法

#### 方法一：使用 xtdatacenter Token 模式（推荐）

适用于**无 QMT 客户端环境**，直接使用 Token 下载数据。

```python
from xtquant import xtdatacenter as xtdc
from xtquant import xtdata
import json
import time

# 1. 读取 Token 配置
with open(".vntrader/vt_setting.json") as f:
    settings = json.load(f)
token = settings.get("datafeed.password", "")

# 2. 初始化 xtdatacenter
xtdc.set_token(token)
xtdc.set_future_realtime_mode(True)  # 期货实时模式
xtdc.init(False)                      # False = 不启动 GUI
xtdc.listen(port=58610)               # 启动本地服务

# 3. 等待服务启动
time.sleep(2)

# 4. 连接 xtdata
xtdata.enable_hello = False
xtdata.connect('127.0.0.1', 58610)

# 5. 下载历史数据到本地缓存
xt_symbol = "p2505.DF"  # 合约代码.交易所（DCE=DF, SHFE=SF, CZCE=ZF）
xtdata.download_history_data(
    stock_code=xt_symbol,
    period='1m',           # 周期: tick, 1m, 5m, 15m, 30m, 1h, 1d
    start_time='20241101', # 开始日期 YYYYMMDD
    end_time='20250430',   # 结束日期 YYYYMMDD
)

# 6. 获取数据
data = xtdata.get_market_data(
    field_list=[],          # 空列表=所有字段
    stock_list=[xt_symbol],
    period='1m',
    start_time='20241101',
    end_time='20250430',
    fill_data=True,         # 填充缺失数据
)

# 7. 数据格式说明
# data 返回格式: {field_name: DataFrame}
# DataFrame 的列是时间戳（如 '20250120213900'），行是股票代码
# 示例: data['open'].loc['p2505.DF'] 获取开盘价序列

# 8. 关闭服务
try:
    xtdc.shutdown()
except Exception:
    pass
```

**交易所代码映射**：

| 交易所 | vnpy Exchange | xtquant 代码 |
|--------|---------------|--------------|
| 大连商品交易所 | DCE | DF |
| 上海期货交易所 | SHFE | SF |
| 郑州商品交易所 | CZCE | ZF |
| 中国金融期货交易所 | CFFEX | IF |
| 上海国际能源交易中心 | INE | INE |

**数据周期**：

| 周期参数 | 说明 |
|----------|------|
| `tick` | Tick 数据 |
| `1m` | 1 分钟 K 线 |
| `5m` | 5 分钟 K 线 |
| `15m` | 15 分钟 K 线 |
| `30m` | 30 分钟 K 线 |
| `1h` | 1 小时 K 线 |
| `1d` | 日 K 线 |

#### 方法二：使用项目封装的下载脚本

项目提供了封装好的下载脚本：

```bash
# 下载棕榈油期货分钟数据
uv run python scripts/download_p2505.py

# 使用 CLI 工具下载（需要 QMT 客户端或配置好 Token）
uv run python -m qp.datafeed.download_palm_oil --symbol p2505 --days 365 --interval 1m --output data/analyse
```

**脚本位置**：
- `scripts/download_p2505.py` - 直接使用 xtdatacenter Token 模式
- `src/qp/datafeed/download_palm_oil.py` - CLI 工具（Tick → K 线合成）
- `src/qp/datafeed/xtquant_feed.py` - XTQuantDatafeed 类封装

#### 方法三：使用 XTQuantDatafeed 类

适用于程序化调用：

```python
from qp.datafeed.xtquant_feed import XTQuantDatafeed
from vnpy.trader.object import HistoryRequest
from vnpy.trader.constant import Exchange, Interval
from datetime import datetime

# 初始化数据源
feed = XTQuantDatafeed()
feed.init()

# 创建请求
req = HistoryRequest(
    symbol="p2505",
    exchange=Exchange.DCE,
    start=datetime(2024, 11, 1),
    end=datetime(2025, 4, 30),
    interval=Interval.MINUTE,
)

# 查询 K 线数据
bars = feed.query_bar_history(req)
print(f"获取到 {len(bars)} 条 K 线数据")

# 查询 Tick 数据
ticks = feed.query_tick_history(req)
print(f"获取到 {len(ticks)} 条 Tick 数据")

feed.close()
```

### 数据缓存

xtquant 下载的数据会缓存在本地：

| 缓存位置 | 说明 |
|----------|------|
| `.venv/Lib/site-packages/xtquant/config/` | xtquant 配置 |
| `data/datadir/DF/60/` | 期货分钟数据缓存（.DAT 文件） |

### 支持的数据周期

| 周期 | 数据源 | 时间跨度 | 说明 |
|------|--------|----------|------|
| Tick | 迅投研 | 无限制 | 最精细粒度 |
| 1分钟 | 迅投研 | 无限制 | 原始分钟数据 |
| 日线 | akshare | ~1年 | `futures_main_sina` |
| 60分钟 | akshare | ~8个月 | `futures_zh_minute_sina(period='60')` |

---

## 11. 数据聚合规范（重要）

### 背景

中国期货交易所（如 DCE 大商所）的主力连续合约数据（如 `p0` 棕榈油）采用**交易时段聚合**方式，每日固定 6 根 K 线。而迅投研等数据源提供的 `p00` 数据采用自然小时聚合（每日 7-8 根）。

**两种聚合方式的回测结果差异显著**，必须统一聚合规范以保证策略可复现性。

### 规范定义

**QuantPlus 标准 K 线聚合规范**：采用交易时段聚合，时间戳为 K 线**结束时间**。

#### DCE 棕榈油交易时段（6 根/天）

| 时段名 | 交易时间 | K 线时间戳 |
|--------|----------|------------|
| 早盘1 | 09:00 - 10:00 | **10:00** |
| 早盘2 | 10:00 - 11:30 | **11:15** |
| 午盘1 | 13:30 - 14:15 | **14:15** |
| 午盘2 | 14:15 - 15:00 | **15:00** |
| 夜盘1 | 21:00 - 22:00 | **22:00** |
| 夜盘2 | 22:00 - 23:00 | **23:00** |

> **注意**：早盘2 包含 10:15-10:30 休息时间，K 线时间戳为 11:15 而非 11:30。

#### 时间戳格式

- **K 线时间戳 = K 线结束时间**（与 p0 主力连续一致）
- 夜盘数据归属**下一个交易日**（如周五夜盘归属周一）

### 数据对比

| 数据类型 | K 线数/天 | 聚合方式 | 时间戳 | 示例 |
|----------|----------|----------|--------|------|
| p0 (主力连续) | 6 | 交易时段 | 结束时间 | 10:00, 11:15, 14:15, 15:00, 22:00, 23:00 |
| p00 (自然小时) | 7-8 | 自然小时 | 开始时间 | 09:00, 10:00, 11:00, 13:00, 14:00, 21:00, 22:00 |
| **qp_session** | 6 | 交易时段 | 结束时间 | **与 p0 一致** |

### SessionBarSynthesizer 使用

```python
from qp.datafeed import SessionBarSynthesizer, DCE_PALM_OIL_SESSIONS

# 创建合成器
synthesizer = SessionBarSynthesizer(sessions=DCE_PALM_OIL_SESSIONS)

# 从分钟数据合成时段 K 线
session_bars = synthesizer.synthesize_from_minutes(minute_bars)

# 从 Tick 数据合成时段 K 线
session_bars = synthesizer.synthesize_from_ticks(tick_data)

# 从时段 K 线合成日 K 线
daily_bars = synthesizer.synthesize_daily(session_bars)
```

### 自定义交易时段

```python
from datetime import time
from qp.datafeed.session_synthesizer import TradingSession, SessionBarSynthesizer

# 定义自定义时段
custom_sessions = [
    TradingSession("时段1", time(9, 0), time(10, 30), time(10, 30)),
    TradingSession("时段2", time(10, 30), time(11, 30), time(11, 30)),
    # ... 更多时段
]

synthesizer = SessionBarSynthesizer(sessions=custom_sessions)
```

### 回测验证

使用 SessionBarSynthesizer 合成的数据与 p0 原始数据回测结果对比：

| 数据源 | 收益率 | Sharpe Ratio | 最大回撤 |
|--------|--------|--------------|----------|
| p0 原始 | +24.41% | 1.28 | -8.23% |
| qp_session 合成 | +27.41% | 1.41 | -7.89% |

> 差异来源：数据获取时间点不同、主力合约换月处理差异等，结果在合理范围内。

---

## 12. 数据获取示例（快速参考）

> 详细说明见第 10 节「数据获取」。

### 迅投研分钟数据（Token 模式）

```bash
# 使用项目封装的下载脚本
uv run python scripts/download_p2505.py
```

### 程序化调用

```python
from qp.datafeed.xtquant_feed import XTQuantDatafeed
from vnpy.trader.object import HistoryRequest
from vnpy.trader.constant import Exchange, Interval
from datetime import datetime

feed = XTQuantDatafeed()
feed.init()

req = HistoryRequest(
    symbol="p2505",
    exchange=Exchange.DCE,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 4, 30),
    interval=Interval.MINUTE,
)
bars = feed.query_bar_history(req)
feed.close()
```

### akshare 备用数据源

```python
import akshare as ak

# 拉取60分钟数据（有限历史深度）
df = ak.futures_zh_minute_sina(symbol='P0', period='60')
print(df.head())
```

---

## 13. 数据清洗规范（重要）

### 问题背景

迅投研下载数据时使用 `fill_data=True` 参数会自动填充非交易时段的数据，导致：

1. **K线数量虚增** - 非交易时段被填充了假K线
2. **技术指标失真** - 零成交K线参与MA/MACD计算
3. **回测结果偏差** - 策略可能在"假K线"上产生信号
4. **K线图错位** - GUI中显示大量平直的假K线

### 问题示例

以 rb2505（螺纹钢）为例，原始下载数据：

| 指标 | 原始数据 | 清理后 |
|------|----------|--------|
| 数据量 | 35,369 条 | 22,080 条 |
| 填充数据占比 | 38% | 0% |
| 零成交K线 | 13,650 条 | 极少 |

填充数据特征：
- 时间戳在 0:00-2:00、23:01-23:59 等非交易时段
- OHLC 四价相同（等于前一交易时段收盘价）
- 成交量和持仓量为 0

### 各交易所交易时段

#### 上期所 SHFE（螺纹钢 rb、热卷等）

| 时段 | 时间 |
|------|------|
| 日盘1 | 09:00 - 10:15 |
| 日盘2 | 10:30 - 11:30 |
| 日盘3 | 13:30 - 15:00 |
| 夜盘 | 21:00 - 23:00 |

#### 大商所 DCE（棕榈油 p、豆粕等）

| 时段 | 时间 |
|------|------|
| 日盘1 | 09:00 - 10:15 |
| 日盘2 | 10:30 - 11:30 |
| 日盘3 | 13:30 - 15:00 |
| 夜盘 | 21:00 - 23:00 |

#### 郑商所 CZCE（纯碱 SA、甲醇 MA、玻璃 FG）

| 时段 | 时间 |
|------|------|
| 日盘1 | 09:00 - 10:15 |
| 日盘2 | 10:30 - 11:30 |
| 日盘3 | 13:30 - 15:00 |
| 夜盘 | 21:00 - 23:00 |

### 数据清洗方法

使用项目提供的清洗脚本：

```bash
# 清洗所有数据
uv run python scripts/clean_trading_hours.py
```

或在代码中手动清洗：

```python
import pandas as pd

def is_trading_time(row):
    """判断是否为交易时段."""
    h, m = row['hour'], row['minute']
    # 09:00-10:15
    if h == 9: return True
    if h == 10 and m <= 15: return True
    # 10:30-11:30
    if h == 10 and m >= 30: return True
    if h == 11 and m <= 30: return True
    # 13:30-15:00
    if h == 13 and m >= 30: return True
    if h == 14: return True
    if h == 15 and m == 0: return True
    # 21:00-23:00
    if h == 21: return True
    if h == 22: return True
    if h == 23 and m == 0: return True
    return False

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['is_trading'] = df.apply(is_trading_time, axis=1)
df_clean = df[df['is_trading']].drop(columns=['hour', 'minute', 'is_trading'])
```

### 实盘 vs 回测数据对比

| 特征 | 填充数据 | 清理后数据 | 实盘数据 |
|------|----------|------------|----------|
| 非交易时段K线 | 有（填充） | 无 | **无** |
| 零成交K线占比 | 高 | 极低 | **极低** |
| K线时间连续性 | 0:00-23:59 | 只有交易时段 | **只有交易时段** |
| 技术指标准确性 | 失真 | 准确 | **准确** |

**结论**：清理后的数据更接近实盘，回测结果更可靠。

### 下载数据时避免填充

在调用 `xtdata.get_market_data()` 时设置 `fill_data=False`：

```python
data = xtdata.get_market_data(
    field_list=[],
    stock_list=[xt_symbol],
    period='1m',
    start_time=start_str,
    end_time=end_str,
    fill_data=False,  # 关键：不填充非交易时段
)
```

---

## 14. 数据归一化与 Session-Aware K线合成

### 问题背景

在使用多数据源（Wind、XTQuant）进行回测对比时，发现分钟级 K 线存在以下不一致问题：

| 问题 | Wind | XTQuant | 影响 |
|------|------|---------|------|
| 时间戳格式 | `:59` 秒（如 `09:04:59`） | `:00` 秒（如 `09:05:00`） | 同一根 bar 时间戳不同，合成窗口错位 |
| Session boundary bars | 可能包含集合竞价 bar | 不包含 | 首根 bar 数据不一致 |
| 零成交噪声 | 可能存在 V=0 且 OHLC 相同的 bar | 较少 | 影响指标计算 |
| 非交易时段数据 | 无 | `fill_data=True` 时会填充 | K 线数量差异大 |

这些差异在逐 bar 增量策略（如 `CtaChanPivotStrategy`）中会被放大：一根 bar 的时间戳偏差 → 窗口归属不同 → 合成的 5m/15m bar 不同 → 指标值不同 → 信号不同 → 回测结果不可复现。

### normalize_1m_bars() 功能

`src/qp/datafeed/normalizer.py` 提供统一的 1 分钟 K 线归一化函数：

```python
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

df_clean = normalize_1m_bars(df_raw, sessions=PALM_OIL_SESSIONS)
```

**处理步骤**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 时间戳标准化 | `:59` 秒 → 下一分钟 `:00`（+1s），统一到分钟精度（`floor("min")`） |
| 2 | 去重 | 同一时间戳保留最后一条（`drop_duplicates(keep="last")`） |
| 3 | 交易时段过滤 | 只保留 `(session_start, session_end]` 区间内的 bar |
| 4 | 零成交噪声剔除 | V=0 且 O=H=L=C 的 bar 视为噪声，移除 |

### SessionSpec 与 PALM_OIL_SESSIONS 定义

```python
@dataclass(frozen=True)
class SessionSpec:
    """交易时段定义."""
    start: time   # 时段开始时间（不含）
    end: time     # 时段结束时间（含）
    name: str = ""
```

**大商所棕榈油 4 个交易时段**：

```python
PALM_OIL_SESSIONS: List[SessionSpec] = [
    SessionSpec(start=time(21, 0), end=time(23, 0), name="night"),   # 夜盘
    SessionSpec(start=time(9, 0),  end=time(10, 15), name="am1"),    # 早盘1
    SessionSpec(start=time(10, 30), end=time(11, 30), name="am2"),   # 早盘2
    SessionSpec(start=time(13, 30), end=time(15, 0),  name="pm"),    # 午盘
]
```

**区间语义**：`(start, end]`，即 start 时刻不含（集合竞价不算），end 时刻包含。

### compute_window_end() 原理

```python
def compute_window_end(dt, sessions, window_minutes) -> Optional[datetime]:
```

**功能**：计算 `dt` 所属的 N 分钟窗口结束时间（session-aware）。

**算法流程**：

```
1. 定位 dt 所属的 session → (session_start, session_end)
2. 计算 dt 距 session_start 的分钟数 elapsed
3. 窗口结束偏移 = ceil(elapsed / window_minutes) × window_minutes
4. window_end = session_start + 窗口结束偏移
5. 截断：if window_end > session_end → window_end = session_end
```

**关键设计**：
- 窗口从 `session_start` 起算对齐，不是从自然整点对齐
- Session 尾部截断：如 am1 时段（09:00, 10:15]，5 分钟窗口最后一个为 10:15 而非 10:20
- 不跨 session：夜盘 23:00 窗口结束后，不会延伸到次日 am1

**示例**（5 分钟窗口，am1 时段）：

| 1m bar 时间戳 | elapsed (分钟) | window_end |
|---------------|---------------|------------|
| 09:01 | 1 | 09:05 |
| 09:05 | 5 | 09:05 |
| 09:06 | 6 | 09:10 |
| 10:11 | 71 | 10:15 |
| 10:15 | 75 | 10:15（尾部截断） |

### get_session_key() 用途

```python
def get_session_key(dt, sessions) -> Optional[Tuple[datetime, datetime]]:
```

返回 `dt` 所属 session 的 `(session_start_dt, session_end_dt)` 元组，作为 session 唯一标识。

**用途**：在策略的 K 线合成中，通过比较 `session_key` 是否变化来判断是否跨 session。跨 session 时必须强制 emit 当前窗口的 bar，避免两个不同 session 的数据被合并到同一根 K 线中。

### 策略中的 Session-Aware 5m/15m 合成

`CtaChanPivotStrategy` 中的 `_update_5m_bar()` / `_update_15m_bar()` 实现了 session-aware 的增量 K 线合成。

**核心逻辑**（以 5m 为例）：

```python
def _update_5m_bar(self, bar: dict) -> Optional[dict]:
    dt = bar['datetime']
    window_end = compute_window_end(dt, self._sessions, 5)
    session_key = get_session_key(dt, self._sessions)
    result = None

    if self._window_bar_5m is not None:
        session_changed = (self._last_session_key_5m != session_key)
        window_changed  = (self._last_window_end_5m != window_end)
        if session_changed or window_changed:
            result = self._window_bar_5m.copy()   # ← emit 上一个窗口
            self._window_bar_5m = None

    # 初始化或更新当前窗口 ...
    return result
```

**关键设计点**：

| 设计 | 说明 |
|------|------|
| 窗口切换 emit | 检测到 `window_end` 变化时，emit 上一个窗口的 bar（而非等 `dt == window_end`） |
| Session 切换 emit | 检测到 `session_key` 变化时，即使窗口未满也强制 emit |
| 先 emit 后更新 | 当前 bar 的数据归入新窗口，不会污染上一个窗口 |
| 返回值语义 | 返回 `None` = 当前窗口仍在累积；返回 `dict` = 上一个窗口已完成 |

**为什么不用 `dt == window_end` 判断？**

不同数据源的尾部 bar 时间戳可能不同（如 Wind 的 10:15 bar 可能是 10:14:59），用精确时间匹配会导致：
- 某些数据源永远不会触发 emit（因为永远没有精确等于 window_end 的 bar）
- 窗口数据被意外合并到下一个窗口

使用「窗口切换」检测则完全避免了这个问题：只要新 bar 属于不同窗口，就 emit 上一个窗口。

### 验证结论

通过 date-aligned 测试验证归一化效果：

1. **测试方法**：取 Wind 和 XTQuant 数据的重叠日期区间，分别经 `normalize_1m_bars()` 处理后，使用 `CtaChanPivotStrategy` 回测
2. **对齐方式**：只保留两个数据源都有数据的交易日（date-aligned），排除数据缺失导致的差异
3. **结果**：在 overlap 区间内，Wind 与 XT 的回测指标**完全一致**（收益率差异 +0.00，Sharpe 差异 +0.00）

> 这证明了归一化 + session-aware 合成方案成功消除了数据源差异，策略结果完全可复现。
