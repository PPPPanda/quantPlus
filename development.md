# Development Guide

本文档提供 quantPlus 项目的开发规范和技术指南。

---

## 1. 项目定位与原则

**本仓库是 quantPlus 主工程**，面向 A 股期货量化交易（主要品种：棕榈油，大商所 DCE）。

### 核心原则

- **vendor/vnpy 只做 submodule + editable install**：原则上不直接修改 vendor/vnpy 源码；如需定制，通过继承/包装在 src/qp 中实现。
- **业务代码只允许放在 src/qp 目录**：禁止在 vendor/vnpy 下开发任何业务逻辑。
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
│       │   └── trader_app.py          # 实盘启动 + GUI 显示入口（支持 profile 参数）
│       ├── strategies/
│       │   └── cta_palm_oil.py        # 双均线策略（棕榈油），先跑起来的基础策略
│       ├── research/
│       │   ├── openbb_fetch.py        # OpenBB 数据拉取 / 因子计算
│       │   └── ingest_vnpy.py         # OpenBB 数据 -> vn.py 数据库/CSV 转换入库
│       └── backtest/
│           └── run_cta_backtest.py    # 脚本化回测入口（批处理/CI 场景）
├── pyproject.toml                     # uv 依赖管理
├── uv.lock                            # 锁定依赖版本
└── development.md                     # 本文件
```

### 职责说明

| 路径 | 职责 |
|------|------|
| `vendor/vnpy/` | vn.py 框架 submodule，只读引用，更新通过 git submodule update |
| `src/qp/runtime/trader_app.py` | GUI 主入口，根据 profile 加载不同 App 组合 |
| `src/qp/strategies/cta_palm_oil.py` | CTA 策略实现，继承 CtaTemplate |
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
# 全功能模式（实盘 + 回测 + 数据管理）
uv run python -m qp.runtime.trader_app --profile all

# 仅投研/回测（加载 CtaBacktesterApp, DataManagerApp）
uv run python -m qp.runtime.trader_app --profile research

# 仅实盘交易（加载 CtaStrategyApp, RiskManagerApp）
uv run python -m qp.runtime.trader_app --profile trade
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
```

### 待 src/qp 模块就绪后可跑

```bash
# Trader GUI 帮助信息
uv run python -m qp.runtime.trader_app --help

# 策略模块导入验证
uv run python -c "from qp.strategies.cta_palm_oil import CtaPalmOilStrategy; print('strategy-ok')"

# 回测脚本帮助
uv run python -m qp.backtest.run_cta_backtest --help
```
