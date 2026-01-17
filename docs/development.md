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
│   └── vnpy/                          # vn.py submodule，独立维护与更新
├── src/
│   └── qp/
│       ├── runtime/
│       │   └── trader_app.py          # 实盘启动 + GUI 显示入口（支持 profile 和 gateway 参数）
│       ├── ui/
│       │   ├── launcher.py            # GUI 启动器
│       │   └── profiles.py            # Profile 配置（trade/research/all）
│       ├── strategies/
│       │   ├── cta_palm_oil.py        # 双均线策略（学习用）
│       │   └── cta_turtle_enhanced.py # 增强海龟策略（推荐）
│       ├── research/
│       │   ├── openbb_fetch.py        # OpenBB 数据拉取 / 因子计算
│       │   └── ingest_vnpy.py         # OpenBB 数据 -> vn.py 数据库/CSV 转换入库
│       └── backtest/
│           └── run_cta_backtest.py    # 脚本化回测入口（批处理/CI 场景）
├── tests/
│   └── test_openctp_connection.py     # OpenCTP 连接测试脚本
├── docs/
│   ├── development.md                 # 本文件
│   ├── openctp_quickstart.md          # OpenCTP 快速上手指南
│   ├── openctp_integration_research.md # OpenCTP 技术调研报告
│   ├── trade_all_features.md          # Trade 模式功能说明
│   ├── quickstart_chart_recorder.md   # K线图和数据录制快速上手
│   └── data_directory_guide.md        # Data 目录说明
├── pyproject.toml                     # uv 依赖管理
└── uv.lock                            # 锁定依赖版本
```

### 职责说明

| 路径 | 职责 |
|------|------|
| `vendor/vnpy/` | vn.py 框架 submodule，只读引用，更新通过 git submodule update |
| `src/qp/runtime/trader_app.py` | GUI 主入口，支持 `--profile` (trade/research/all) 和 `--gateway` (ctp/tts) 参数 |
| `src/qp/strategies/cta_palm_oil.py` | 双均线 CTA 策略，继承 CtaTemplate（学习用） |
| `src/qp/strategies/cta_turtle_enhanced.py` | 增强海龟策略（趋势过滤+中轨止盈）**推荐** |
| `src/qp/research/openbb_fetch.py` | 研究层数据获取，与交易运行时隔离 |
| `src/qp/research/ingest_vnpy.py` | 数据桥接：将 OpenBB 数据转为 vn.py 可用格式 |
| `src/qp/backtest/run_cta_backtest.py` | 脚本化回测，用于批量参数优化或 CI 流水线；图形化回测推荐走 GUI 的 research profile |

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
from vnpy.trader.utility import ArrayManager
from vnpy_ctastrategy import CtaTemplate

class MyStrategy(CtaTemplate):
    author = "QuantPlus"

    # 参数（GUI 可见）
    param1: int = 10
    parameters = ["param1"]

    # 变量（GUI 可见）
    var1: float = 0.0
    variables = ["var1"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        # ArrayManager size 必须覆盖所有窗口 + buffer
        self.am = ArrayManager(size=self.param1 + 20)

    def on_init(self):
        self.load_bar(10)  # 预加载历史数据

    def on_bar(self, bar):
        self.am.update_bar(bar)
        if not self.am.inited:
            return  # warm-up 未完成，不交易
        # 交易逻辑...
```

### 工程最佳实践

| 要点 | 说明 |
|------|------|
| **Warm-up 防护** | `ArrayManager size = max(所有窗口) + 20`，防止 0 成交 |
| **Look-ahead 防护** | 计算指标时使用 `[-window-1:-1]` 切片，避免使用当前 bar |
| **止损单管理** | 每根 bar 调用 `cancel_all()` 后重新挂止损单 |
| **日志记录** | 入场/出场/加仓必须 `write_log()`，便于调试 |
| **参数校验** | `on_init()` 中校验参数合法性 |

### GUI 可见性

策略需要放置桥接文件才能在 GUI 中显示：

```bash
# .vntrader/strategies/{策略文件名}.py
from qp.strategies.{策略模块} import {策略类名}
```

### 回测验证

新策略必须通过脚本化回测验证：

```bash
uv run python -m qp.backtest.run_cta_backtest --strategy {策略类名} --days 365 -v
```

---

## 9. 现有策略说明

### 策略回测对比（60分钟数据，2025-05 ~ 2026-01，约8个月）

| 策略 | Sharpe Ratio | 总收益率 | 年化收益 | 最大回撤 | 推荐度 |
|------|-------------|---------|---------|---------|--------|
| CtaTurtleEnhancedStrategy (激进) | **1.28** | **+24.41%** | **+34.07%** | -8.23% | **推荐** |
| CtaPalmOilStrategy | -1.50 | -1.64% | -2.28% | -1.85% | 学习用 |

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

## 10. 数据获取

### 数据源

| 数据源 | 用途 | 说明 |
|--------|------|------|
| 迅投研 (XTQuant) | **主数据源** | Tick/分钟数据，支持 Token 连接 |
| akshare (新浪) | 备用 | 有限历史深度，无 Tick |

### 迅投研配置

使用 vnpy_xt 包进行数据获取，支持 Token 连接（无需 QMT 客户端）：

```python
# 配置 settings.json
{
    "datafeed.name": "xt",
    "datafeed.username": "client",  # Token 模式
    "datafeed.password": "<your_token>"
}
```

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

## 12. 数据获取示例

### 迅投研分钟数据（推荐）

```python
from vnpy_xt import Datafeed
from vnpy.trader.object import HistoryRequest
from vnpy.trader.constant import Exchange, Interval
from datetime import datetime

# 初始化
datafeed = Datafeed()
datafeed.init()

# 查询分钟数据
req = HistoryRequest(
    symbol="p00",
    exchange=Exchange.DCE,
    start=datetime(2024, 1, 1),
    end=datetime(2025, 1, 1),
    interval=Interval.MINUTE
)
bars = datafeed.query_bar_history(req)
```

### akshare 备用数据源

```python
import akshare as ak

# 拉取60分钟数据
df = ak.futures_zh_minute_sina(symbol='P0', period='60')
print(df.head())
```
