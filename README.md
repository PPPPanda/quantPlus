# quantPlus

A 股期货量化交易工程，基于 [VeighNa (vnpy)](https://github.com/vnpy/vnpy) 框架构建。

主要品种：棕榈油（大商所 DCE）

## 特性

- **模块化架构**：策略、回测、研究、运行时分层解耦
- **双数据源**：OpenBB 用于研究分析，vn.py 数据库用于实盘/回测
- **GUI 支持**：VeighNa Trader 图形界面，支持实盘交易、回测、数据管理
- **Profile 模式**：trade（实盘）、research（投研回测）、all（全功能）
- **uv 管理**：统一依赖锁定，可复现环境

## 环境要求

- Python 3.11+（推荐 3.13）
- [uv](https://github.com/astral-sh/uv) 包管理器
- 桌面环境（GUI 模式需要）

## 安装

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/yourname/quantPlus.git
cd quantPlus
```

### 2. 安装依赖

**完整安装（GUI + 交易 + 研究）**

```bash
uv sync --extra gui --extra trade --extra research
```

**带 GUI（本地开发/交易）**

```bash
uv sync --extra gui --extra trade
```

**无 GUI（服务器/CI）**

```bash
uv sync --extra trade
```

**仅研究（数据分析）**

```bash
uv sync --extra research
```

## 运行

### Trader GUI

```bash
# 全功能模式（实盘 + 回测 + 数据管理）
uv run python -m qp.runtime.trader_app --profile all

# 投研/回测模式（CtaBacktester + DataManager）
uv run python -m qp.runtime.trader_app --profile research

# 实盘交易模式（CtaStrategy + RiskManager）
uv run python -m qp.runtime.trader_app --profile trade

# 查看帮助
uv run python -m qp.runtime.trader_app --help
```

### Profile 说明

| Profile | 用途 | 加载的 App |
|---------|------|-----------|
| `trade` | 实盘交易 | CtaStrategy, RiskManager, (DataRecorder) |
| `research` | 投研回测 | CtaBacktester, DataManager, (ChartWizard) |
| `all` | 全功能调试 | 以上全部 + PaperAccount |

注：括号内为可选模块，依赖安装情况自动加载。

## 棕榈油（DCE）一年数据回测（GUI）

完整的数据拉取 → 入库 → GUI 回测闭环操作指南。

### 1. 安装依赖

```bash
uv sync --extra gui --extra trade --extra research
```

### 2. 拉取数据并入库

一键流水线命令：

```bash
uv run python -m qp.research.pipeline_palm_oil --vt_symbol p0.DCE --days 365
```

此命令会：
- 使用 akshare 获取棕榈油连续合约 (p0) 近一年日线数据
- 将数据写入 `.vntrader/database.db`
- 输出 GUI 回测操作指引

### 3. 启动 GUI

```bash
uv run python -m qp.runtime.trader_app --profile research
```

### 4. 在 GUI 中运行回测

1. **打开回测界面**
   - 菜单栏：`功能` → `CTA回测`

2. **选择策略**
   - 在「交易策略」下拉框中选择：`CtaPalmOilStrategy`
   - 如果看不到该策略，点击「策略重载」按钮刷新

3. **配置回测参数**
   | 参数 | 值 | 说明 |
   |------|-----|------|
   | 本地代码 | `p0.DCE` | 完整的 vt_symbol 格式 |
   | K线周期 | `d` | 日线 |
   | 开始日期 | 2025-01-15 | 数据起始日期（约一年前） |
   | 结束日期 | 2026-01-13 | 数据结束日期 |
   | 手续费率 | `0.0001` | 万分之一 |
   | 交易滑点 | `2` | 棕榈油最小变动价位 |
   | 合约乘数 | `10` | 棕榈油合约乘数 |
   | 价格跳动 | `2` | 棕榈油最小变动价位 |
   | 回测资金 | `1000000` | 初始资金 |

4. **运行回测**
   - 点击「开始回测」按钮
   - 等待回测完成后查看：
     - 右侧统计指标面板
     - 资金曲线图
     - 「成交记录」「每日盈亏」等详细数据

### 5. 数据管理

查看已入库的数据：
- 菜单栏：`功能` → `数据管理`
- 可查看、导出 p0.DCE 的历史数据

### 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| GUI 看不到数据 | 未在仓库根目录运行 | 确保在 `quantPlus/` 目录下执行命令 |
| 回测无数据 | vt_symbol 不一致 | 确保本地代码填 `p0.DCE`（完整格式） |
| **回测 0 成交** | **ArrayManager 预热数据不足** | **见下方"策略预热与数据量"说明** |
| 数据库不存在 | 未运行入库流水线 | 执行 `pipeline_palm_oil` 命令 |
| akshare 报错 | 网络问题或 API 变更 | 检查网络，或更新 akshare 版本 |

### 策略预热与数据量

**重要**：CTA 策略使用 `ArrayManager` 计算技术指标，需要累积足够的 K 线数据才能开始交易。

**原理**：
- `ArrayManager(size=N)` 需要累积 N 根 K 线后 `am.inited` 才变为 True
- 只有 `am.inited=True` 后，策略才会计算均线并产生交易信号
- 如果回测数据量 ≤ ArrayManager size，策略将永远不会产生交易

**CtaPalmOilStrategy 默认配置**：
- `slow_window = 20`
- `ArrayManager size = slow_window + 10 = 30`
- **最小数据量要求：> 30 条**（推荐使用一年数据，约 240 条）

**如何验证**：
```bash
# 脚本化回测验证（不依赖 GUI）
uv run python -m qp.backtest.run_cta_backtest --vt_symbol p0.DCE --days 365
```

**典型问题场景**：
- 回测 61 条数据，ArrayManager size=60 → 只有最后 1-2 根 K 线能产生信号 → 0 成交
- 修复：将 ArrayManager size 改为 `slow_window + 10`（30），数据中有足够 K 线可用于交易

### 单独执行各步骤

如需手动控制每个步骤：

```bash
# 1. 仅拉取数据（保存到 CSV）
uv run python -m qp.research.openbb_fetch --vt_symbol p0.DCE --days 365

# 2. 仅入库（从 CSV 写入数据库）
uv run python -m qp.research.ingest_vnpy \
    --csv data/openbb/p0.DCE_1d.csv \
    --vt_symbol p0.DCE \
    --interval DAILY

# 3. 验证数据库
uv run python -c "
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
from datetime import datetime, timedelta
db = get_database()
bars = db.load_bar_data('p0', Exchange.DCE, Interval.DAILY,
                        datetime.now()-timedelta(days=400), datetime.now())
print(f'数据条数: {len(bars)}')
if bars:
    print(f'最新: {bars[-1].datetime} C={bars[-1].close_price}')
"
```

## 项目结构

```
quantPlus/
├── vendor/vnpy/           # vn.py 框架 (submodule, 只读)
├── src/qp/
│   ├── ui/                # GUI 启动器与 Profile 配置
│   ├── runtime/           # 命令行入口
│   ├── strategies/        # CTA 策略实现
│   │   └── cta_palm_oil.py    # 双均线策略
│   ├── research/          # 数据研究与入库
│   │   ├── openbb_fetch.py    # 数据拉取（OpenBB/akshare）
│   │   ├── ingest_vnpy.py     # 数据入库
│   │   └── pipeline_palm_oil.py # 一键流水线
│   └── backtest/          # 脚本化回测
│       └── run_cta_backtest.py # 命令行回测工具
├── .vntrader/             # vn.py 数据目录（本地）
│   └── database.db        # SQLite 历史数据库
├── data/openbb/           # 数据缓存（CSV）
├── pyproject.toml         # uv 依赖配置
├── development.md         # 开发规范
└── README.md
```

## 合约代码规范

格式：`合约代码.交易所`

示例：`p2405.DCE`（棕榈油 2405，大商所）

- `DCE` = 大连商品交易所
- `p` = 棕榈油品种代码

## Smoke Tests

验证安装是否正确：

```bash
# 1. 验证 vnpy 导入
uv run python -c "import vnpy; print('vnpy-ok')"

# 2. 验证 OpenBB 导入
uv run python -c "from openbb import obb; print('openbb-ok')"

# 3. 验证 PySide6/Qt（需要 --extra gui）
uv run python -c "from PySide6.QtWidgets import QApplication; print('pyside6-ok')"

# 4. 验证 Trader 入口
uv run python -m qp.runtime.trader_app --help

# 5. 验证策略模块
uv run python -c "from qp.strategies.cta_palm_oil import CtaPalmOilStrategy; print('strategy-ok')"

# 6. 验证数据拉取模块
uv run python -m qp.research.openbb_fetch --help

# 7. 验证数据入库模块
uv run python -m qp.research.ingest_vnpy --help

# 8. 验证脚本化回测
uv run python -m qp.backtest.run_cta_backtest --help
```

## 常见问题

### Q: 启动时提示 "No module named 'vnpy_spreadtrading'"

A: 这是可选依赖，ChartWizard 和 DataRecorder 会自动跳过。如需启用，手动安装：

```bash
uv add vnpy-spreadtrading
```

### Q: Windows 下中文乱码

A: 终端编码问题，不影响功能。可尝试设置 `chcp 65001` 或使用 Windows Terminal。

## 文档

详细开发规范请参阅 [development.md](./development.md)

## License

MIT
