# Trade All 模式功能说明

## 概述

`trade --profile all` 模式是**全功能模式**，集成了实盘交易、投研回测、K线图分析和数据录制等全部功能。

## 启动方式

```bash
# 启动实盘交易模式（包含K线图）
uv run python -m qp.runtime.trader_app --profile trade

# 启动全功能模式（实盘+回测+分析）
uv run python -m qp.runtime.trader_app --profile all

# 简化命令（默认就是all模式）
uv run python -m qp.runtime.trader_app
```

## 模式对比

| 模式 | 模块数量 | 包含功能 | 适用场景 |
|------|---------|---------|---------|
| **trade** | 4个 | CTA策略 + 数据录制 + 风控 + **K线图** | 实盘交易 |
| **research** | 3个 | CTA回测 + 数据管理 + K线图 | 策略研发 |
| **all** | 7个 | trade + research + 模拟盘 | 全功能开发 |

---

## 已集成模块列表

### 1. 实盘交易模块

#### CtaStrategyApp - CTA策略引擎
**功能**：
- 实盘策略运行和管理
- 策略参数动态调整
- 持仓和委托监控
- 策略日志查看

**使用方法**：
1. 菜单：`功能` → `CTA策略`
2. 加载策略文件（从 `strategies/` 目录）
3. 设置参数、初始化、启动

#### RiskManagerApp - 风控管理
**功能**：
- 单笔最大委托量限制
- 总持仓限制
- 单日最大撤单次数限制
- 流控管理（防止频繁下单）

**使用方法**：
1. 菜单：`功能` → `风控管理`
2. 设置风控参数
3. 启动风控引擎

---

### 2. K线图分析模块 ✨

#### ChartWizardApp - 图表向导
**功能**：
- **实时K线图显示**（1分钟/5分钟/日线等）
- **技术指标叠加**（MA/MACD/RSI/布林带等）
- **历史K线回放**
- **多周期联动分析**
- **画线工具**（趋势线、水平线、通道等）

**使用方法**：
1. 菜单：`功能` → `K线图表`
2. 输入合约代码（如 `p2505.DCE` 棕榈油2025年5月）
3. 选择时间周期（1分钟/5分钟/15分钟/1小时/日线）
4. 添加技术指标

**快捷键**：
- `Ctrl+鼠标滚轮`：缩放K线
- `左键拖动`：平移图表
- `右键`：画线工具菜单

**支持的技术指标**：
```
均线系统：
  - SMA（简单移动平均）
  - EMA（指数移动平均）
  - 布林带（Bollinger Bands）

趋势指标：
  - MACD（平滑异同移动平均）
  - DMI（动向指标）
  - ADX（平均趋向指数）

震荡指标：
  - RSI（相对强弱指标）
  - KDJ（随机指标）
  - CCI（顺势指标）

成交量：
  - Volume（成交量柱状图）
  - OBV（能量潮）
```

**示例配置**：
```python
# 查看棕榈油主力合约日K线
合约代码: p00.DCE
周期: 1天
指标: MA(5,10,20) + MACD + Volume

# 查看螺纹钢1小时K线
合约代码: rb2505.SHFE
周期: 1小时
指标: EMA(12,26) + RSI(14)
```

---

### 3. 数据自动录制模块 ✨

#### DataRecorderApp - 数据记录器
**功能**：
- **实时Tick数据录制**（逐笔行情）
- **实时K线数据录制**（1分钟/5分钟等）
- **自动保存到数据库**（SQLite/MongoDB）
- **断线重连自动补录**
- **数据完整性校验**

**使用方法**：
1. 菜单：`功能` → `数据记录`
2. 添加要录制的合约
3. 选择录制类型（Tick/K线/全部）
4. 启动录制

**录制配置**：
```python
# 在GUI中配置
合约列表：
  p2505.DCE    # 棕榈油2025年5月
  rb2505.SHFE  # 螺纹钢2025年5月
  i2505.DCE    # 铁矿石2025年5月

录制类型：
  ☑ Tick数据
  ☑ 1分钟K线
  ☑ 5分钟K线

存储位置：
  .vntrader/database.db (默认SQLite)
```

**数据存储格式**：
```sql
-- Tick表结构
CREATE TABLE tick_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    exchange TEXT,
    datetime TIMESTAMP,
    last_price REAL,
    volume REAL,
    open_interest REAL,
    bid_price_1 REAL,
    ask_price_1 REAL,
    ...
);

-- K线表结构
CREATE TABLE bar_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    exchange TEXT,
    datetime TIMESTAMP,
    interval TEXT,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume REAL,
    ...
);
```

**数据导出**：
```python
# 使用DataManagerApp导出数据
from vnpy.trader.database import get_database

db = get_database()
bars = db.load_bar_data(
    symbol="p2505",
    exchange=Exchange.DCE,
    interval=Interval.MINUTE,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 1, 31)
)

# 导出为CSV
import pandas as pd
df = pd.DataFrame([{
    "datetime": bar.datetime,
    "open": bar.open_price,
    "high": bar.high_price,
    "low": bar.low_price,
    "close": bar.close_price,
    "volume": bar.volume
} for bar in bars])
df.to_csv("p2505_202501.csv", index=False)
```

---

### 4. 投研回测模块

#### CtaBacktesterApp - CTA回测引擎
**功能**：
- 历史数据回测
- 参数优化（网格搜索）
- 回测报告生成
- 绩效指标分析

#### DataManagerApp - 数据管理器
**功能**：
- 历史数据下载（多数据源）
- 数据导入/导出
- 数据清洗和修复
- 数据统计分析

---

### 5. 模拟账户

#### PaperAccountApp - 模拟盘
**功能**：
- 实盘行情 + 模拟交易
- 无需真实资金
- 测试策略逻辑
- 积累实战经验

---

## 典型使用场景

### 场景1：实时监控 + K线分析

```
1. 启动 trade --profile all
2. 连接CTP（菜单：系统 → 连接CTP）
3. 打开K线图（功能 → K线图表）
4. 输入合约代码查看实时K线
5. 叠加技术指标辅助决策
```

### 场景2：策略实盘 + 数据录制

```
1. 启动 trade --profile all
2. 连接CTP
3. 启动数据录制（功能 → 数据记录）
   - 添加主力合约
   - 启动Tick+1分钟录制
4. 加载CTA策略（功能 → CTA策略）
5. 策略运行 + 数据持续录制
```

### 场景3：回测 + 实盘验证

```
1. 使用DataManagerApp下载历史数据
2. 使用CtaBacktesterApp回测策略
3. 参数优化找到最佳参数
4. 切换到CtaStrategyApp实盘运行
5. 用ChartWizardApp监控实盘表现
```

---

## 数据录制最佳实践

### 1. 选择录制品种

**建议录制**：
- 主力合约（如 `p00`, `rb00`, `i00`）
- 活跃次主力（如 `p2505`, `rb2505`）
- 套利对冲品种

**不建议录制**：
- 远月非活跃合约（成交量<100手/天）
- 已下市合约

### 2. 录制周期配置

```
Tick数据：适合高频策略开发
  - 存储量大（1个品种1天≈50MB）
  - 精度最高（毫秒级）

1分钟K线：适合日内策略
  - 存储量小（1个品种1天≈10KB）
  - 精度足够（大部分策略）

5分钟/15分钟：适合趋势策略
  - 存储量更小
  - 过滤短期噪音
```

### 3. 存储管理

```bash
# 定期备份数据库
cp .vntrader/database.db backups/database_$(date +%Y%m%d).db

# 清理旧数据（保留近1年）
sqlite3 .vntrader/database.db "
DELETE FROM tick_data WHERE datetime < datetime('now', '-365 days');
DELETE FROM bar_data WHERE datetime < datetime('now', '-365 days');
VACUUM;
"

# 检查数据库大小
du -h .vntrader/database.db
```

---

## 性能优化建议

### K线图性能

```python
# 限制历史K线加载数量
最大K线数: 10000根（约40天1分钟K线）

# 减少技术指标数量
推荐: 3个以内（如 MA + MACD + Volume）

# 使用日线/小时线代替分钟线（长周期分析）
```

### 数据录制性能

```python
# 高频录制（20个品种以上）建议配置
数据库: MongoDB（比SQLite更快）
存储: SSD硬盘
内存: ≥8GB

# 网络优化
使用7x24服务器（避免夜盘断线）
备用连接（主备切换）
```

---

## 常见问题

### Q1: K线图显示空白？

**原因**：数据库中没有该合约的历史数据

**解决**：
1. 使用DataManagerApp下载历史数据
2. 或连接实盘后等待实时K线生成
3. 或启动DataRecorder录制后查看

### Q2: 数据录制不启动？

**原因**：未连接行情网关

**解决**：
1. 先连接CTP（系统 → 连接CTP）
2. 确认行情连接成功（状态栏显示"已连接"）
3. 再启动数据录制

### Q3: 数据库文件过大怎么办？

**解决**：
```bash
# 方案1：删除旧数据
DELETE FROM tick_data WHERE datetime < '2024-01-01';

# 方案2：导出后重建数据库
# 导出CSV → 删除database.db → 重新录制

# 方案3：使用MongoDB（支持TB级数据）
```

---

## 模块依赖

```toml
# pyproject.toml
dependencies = [
    "vnpy-ctastrategy>=1.0.0",    # CTA策略引擎
    "vnpy-datarecorder>=1.1.0",   # 数据录制器 ✨
    "vnpy-chartwizard>=1.1.0",    # K线图表 ✨
    "vnpy-riskmanager>=1.0.0",    # 风控管理
    "vnpy-ctabacktester>=1.0.0",  # CTA回测
    "vnpy-datamanager>=1.0.0",    # 数据管理
    "vnpy-paperaccount>=1.0.0",   # 模拟盘
]
```

---

## 扩展阅读

- [VeighNa官方文档](https://www.vnpy.com/docs/)
- [CTA策略开发指南](https://www.vnpy.com/docs/cn/cta_strategy.html)
- [数据录制器使用说明](https://www.vnpy.com/docs/cn/data_recorder.html)
- [K线图表使用教程](https://www.vnpy.com/docs/cn/chart_wizard.html)

---

*文档更新时间：2026-01-17*
