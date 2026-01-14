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

## 数据获取

### 支持的数据周期

| 周期 | 数据条数 | 时间跨度 | 用途 |
|------|----------|----------|------|
| 日线 (DAILY) | ~365 | 约1年 | 中长期策略回测 |
| 60分钟 (HOUR) | ~1023 | 约8个月 | 日内/短期策略回测 |
| 5分钟 | ~1023 | 约16个交易日 | 高频策略研究 |
| 1分钟 | ~1023 | 约4个交易日 | 高频策略研究 |

数据来源：akshare (新浪财经)

### 拉取60分钟数据（推荐）

```bash
# 拉取60分钟数据并保存为CSV
uv run python -c "
import akshare as ak
import pandas as pd
from pathlib import Path

data_dir = Path('data/openbb')
data_dir.mkdir(parents=True, exist_ok=True)

df = ak.futures_zh_minute_sina(symbol='P0', period='60')
df = df.rename(columns={'hold': 'open_interest'})
df.to_csv(data_dir / 'p0.DCE_1h.csv', index=False)
print(f'数据已保存，共 {len(df)} 条')
"

# 入库到 vn.py
uv run python -m qp.research.ingest_vnpy --csv data/openbb/p0.DCE_1h.csv --vt_symbol p0.DCE --interval HOUR
```

### 拉取日线数据

一键流水线命令：

```bash
uv run python -m qp.research.pipeline_palm_oil --vt_symbol p0.DCE --days 365
```

此命令会：
- 使用 akshare 获取棕榈油连续合约 (p0) 近一年日线数据
- 将数据写入 `.vntrader/database.db`
- 输出 GUI 回测操作指引

## 棕榈油（DCE）回测（GUI）

完整的数据拉取 → 入库 → GUI 回测闭环操作指南。

### 1. 安装依赖

```bash
uv sync --extra gui --extra trade --extra research
```

### 2. 启动 GUI

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

## 策略说明

### 策略回测对比（60分钟数据，2025-05 ~ 2026-01，约8个月）

| 策略 | Sharpe Ratio | 总收益率 | 年化收益 | 最大回撤 | 推荐度 |
|------|-------------|---------|---------|---------|--------|
| CtaTurtleEnhancedStrategy (激进) | **1.28** | **+24.41%** | **+34.07%** | -8.23% | **推荐** |
| CtaPalmOilStrategy | -1.50 | -1.64% | -2.28% | -1.85% | 学习用 |

**结论**：增强海龟策略经过激进优化后，收益率达到24%+，年化34%，远超目标。

### CtaTurtleEnhancedStrategy（增强海龟策略）**推荐**

改进版海龟策略，针对高收益优化，特性：
- **趋势过滤**：MA10 > MA100 时只做多，MA10 < MA100 时只做空（超长周期过滤噪音）
- **中轨止盈**：触及唐奇安通道中线时止盈，锁定利润
- **极速出场**：exit_window=1，最快响应反转信号
- **激进仓位**：risk_per_trade=6%，max_units=50
- **快速加仓**：pyramid_atr=0.15，利润快速累积

**默认参数**（2026-01 激进优化: Sharpe 1.28, 收益 +24.41%, 年化 +34.07%）:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `entry_window` | 15 | 入场通道窗口 |
| `exit_window` | 1 | 出场通道窗口（极速出场） |
| `trend_ma_fast` | 10 | 快速趋势均线 |
| `trend_ma_slow` | 100 | 慢速趋势均线（超长周期过滤） |
| `atr_stop` | 1.5 | ATR 止损倍数 |
| `risk_per_trade` | 0.06 | 单笔风险预算 6% |
| `max_units` | 50 | 最大持仓手数 |
| `pyramid_atr` | 0.15 | 加仓间隔 ATR 倍数 |
| `use_trend_filter` | True | 趋势过滤开关 |
| `use_mid_line_exit` | True | 中轨止盈开关 |
| `enable_pyramid` | True | 金字塔加仓开关 |

**保守版参数**（Sharpe 1.14, 收益 +6.85%, 回撤 -2.92%）:
```python
# 适合风险偏好较低的用户
setting = {
    'exit_window': 2,
    'trend_ma_slow': 80,
    'risk_per_trade': 0.03,
    'max_units': 15,
    'pyramid_atr': 0.5,
}
```

#### GUI 回测完整流程（60分钟数据）

**第一步：准备60分钟数据**

```bash
# 1. 拉取60分钟数据
uv run python -c "
import akshare as ak
from pathlib import Path

data_dir = Path('data/openbb')
data_dir.mkdir(parents=True, exist_ok=True)

df = ak.futures_zh_minute_sina(symbol='P0', period='60')
df = df.rename(columns={'hold': 'open_interest'})
df.to_csv(data_dir / 'p0.DCE_1h.csv', index=False)
print(f'数据已保存，共 {len(df)} 条')
"

# 2. 入库到 vn.py（注意 interval 参数为 HOUR）
uv run python -m qp.research.ingest_vnpy --csv data/openbb/p0.DCE_1h.csv --vt_symbol p0.DCE --interval HOUR
```

**第二步：启动 GUI**

```bash
uv run python -m qp.runtime.trader_app --profile research
```

**第三步：配置回测参数**

菜单栏：`功能` → `CTA回测`，按下表配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| **交易策略** | `CtaTurtleEnhancedStrategy` | 在下拉框中选择，看不到点「策略重载」 |
| **本地代码** | `p0.DCE` | 必须与入库时的 vt_symbol 一致 |
| **K线周期** | `1h` | **重要**：60分钟数据必须填 `1h`，不是 `60` |
| 开始日期 | `2025-05-07` | 数据起始日期 |
| 结束日期 | `2026-01-14` | 数据结束日期 |
| 手续费率 | `0.0001` | 万分之一 |
| 交易滑点 | `2` | 棕榈油最小变动价位 |
| 合约乘数 | `10` | 棕榈油合约乘数 |
| 价格跳动 | `2` | 棕榈油最小变动价位 |
| 回测资金 | `1000000` | 初始资金 |

**K线周期对照表**：

| 数据周期 | K线周期填写 | interval 参数 |
|----------|-------------|---------------|
| 日线 | `d` 或 `1d` | `DAILY` |
| 60分钟 | `1h` | `HOUR` |
| 30分钟 | `30m` | - |
| 15分钟 | `15m` | - |
| 5分钟 | `5m` | - |
| 1分钟 | `1m` | `MINUTE` |

**第四步：修改策略参数（可选）**

如需使用保守版参数：

1. 在回测界面右侧找到「策略参数」区域
2. 点击「参数设置」按钮
3. 在弹出窗口中修改参数：
   ```
   exit_window: 2
   trend_ma_slow: 80
   risk_per_trade: 0.03
   max_units: 15
   pyramid_atr: 0.5
   ```
4. 点击「确定」保存

**第五步：运行回测**

1. 点击「开始回测」按钮
2. 等待进度条完成
3. 查看结果：
   - **统计指标**：右侧面板显示 Sharpe、收益率、最大回撤等
   - **资金曲线**：点击「K线图表」查看资金变化
   - **成交记录**：点击「成交记录」查看每笔交易详情
   - **每日盈亏**：点击「每日盈亏」查看每日收益分布

**常见问题**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 回测 0 成交 | K线周期设置错误 | 60分钟数据必须填 `1h`，不是 `60` 或 `h` |
| 找不到策略 | 桥接文件缺失 | 点击「策略重载」或检查 `.vntrader/strategies/` |
| 数据量为 0 | vt_symbol 不匹配 | 确保本地代码填 `p0.DCE`（区分大小写） |
| ArrayManager 未初始化 | 数据不足 | 该策略需要 > 130 条数据（trend_ma_slow=100 + buffer） |

**命令行回测**（不依赖 GUI）：
```bash
uv run python -m qp.backtest.run_cta_backtest --strategy CtaTurtleEnhancedStrategy --interval HOUR --days 240
```

### CtaPalmOilStrategy（双均线策略）

基础趋势跟踪策略，适合入门学习。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fast_window` | 10 | 快速均线周期 |
| `slow_window` | 20 | 慢速均线周期 |
| `fixed_size` | 1 | 每次交易手数 |

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
│   │   ├── cta_palm_oil.py            # 双均线策略（学习用）
│   │   └── cta_turtle_enhanced.py     # 增强海龟策略（推荐）
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

# 6. 验证增强海龟策略模块
uv run python -c "from qp.strategies.cta_turtle_enhanced import CtaTurtleEnhancedStrategy; print('turtle-ok')"

# 7. 验证数据拉取模块
uv run python -m qp.research.openbb_fetch --help

# 8. 验证数据入库模块
uv run python -m qp.research.ingest_vnpy --help

# 9. 验证脚本化回测
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
