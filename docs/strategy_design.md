# CtaChanPivotStrategy 策略设计说明文档

> 版本：iter8 final | 日期：2026-02-05 | 代码：`src/qp/strategies/cta_chan_pivot.py` (~1500行)

---

## 一、策略总览

### 1.1 一句话描述
基于**缠论中枢(ZhongShu)**的日内趋势跟随策略，在 1 分钟 K 线上运行，内部合成 5m/15m K线，通过识别中枢的形成→离开→回试结构产生买卖信号，配合 ATR 移动止损和多层风控系统管理仓位。

### 1.2 核心信号

| 信号 | 类型 | 触发条件 | 缠论依据 |
|------|------|----------|----------|
| **3B** | 主买点 | 价格向上离开中枢 → 回踩不破 ZG → 15m 多头 | 缠论第三类买点 |
| **3S** | 平多/卖点 | 价格向下离开中枢 → 回抽不破 ZD → 15m 空头 | 缠论第三类卖点 |
| **2B** | 辅助买点 | 中枢内低点抬高 + MACD动能增强 + 15m 多头 | 缠论第二类买点（简化） |
| **2S** | 辅助平多 | 中枢内高点降低 + MACD动能减弱 + 15m 空头 | 缠论第二类卖点（简化） |
| **S6底背驰** | 补充买点 | 价格创新低 + MACD面积衰减 + 中枢存在 | 缠论背驰 |

### 1.3 约束
- **只做多，不做空**（iter4 教训：做空在棕榈油上灾难性）
- 3S/2S 仅用于**平仓**，不反手开空
- 同一时刻最多持 1 手

### 1.4 适用标的
- 大商所棕榈油期货（p系列合约）
- 交易时段：夜盘 21:00-23:00，日盘 9:00-10:15 / 10:30-11:30 / 13:30-15:00

---

## 二、数据处理流水线

### 2.1 数据流
```
交易所 1m Tick/Bar
    ↓
[归一化] normalize_1m_bars()          ← 仅回测时，过滤非交易时段bar
    ↓
策略 on_bar()
    ├── Session 过滤（非交易时段 bar 直接跳过）
    ├── 1m 级别：止损检查 + 入场执行
    ├── → _update_15m_bar() → 15m MACD（趋势过滤）
    └── → _update_5m_bar()  → 5m K线处理链：
         ├── 包含处理 _process_inclusion()
         ├── 分型识别 + 严格笔 _process_bi()
         ├── MACD/ATR 增量计算
         ├── 中枢状态机 _update_pivots()
         └── 信号检查 _check_signal()
```

### 2.2 K线合成（Session-Aware）
策略不使用外部指标库，完全**增量计算**：
- 5m/15m K线合成使用 `compute_window_end()` 进行 session-aware 窗口对齐
- 窗口在 session 边界强制截断（不跨 session）
- session 切换时强制 emit 上一个累积窗口

### 2.3 归一化说明
`normalize_1m_bars()` 对 CSV 原始数据做三件事：
1. 时间戳标准化（:59秒 → :00，去重）
2. 过滤非交易时段 bar
3. 剔除零成交噪声 bar（V=0 且 O=H=L=C）

策略内部也有 session 过滤（`on_bar()` 开头检查），确保即使数据库有非交易时段数据也能正确运行。

---

## 三、缠论结构实现

### 3.1 包含处理 `_process_inclusion()`
- 三根K线（left-mid-curr），mid 与 curr 存在包含关系时合并
- 包含方向 `_inclusion_dir`：向上取高高低高，向下取低高低低
- **已知偏差**：初始方向默认向上（`dir=0→1`），前几根K线可能不准，后续自然纠正
- 输出：包含处理后的 K 线列表 `_k_lines`

### 3.2 分型与严格笔 `_process_bi()`
- **顶分型**：mid.high > left.high 且 mid.high > curr.high
- **底分型**：mid.low < left.low 且 mid.low < curr.low
- **严格笔规则**：异向分型间距 ≥ `min_bi_gap=4`（近似缠论5根K线要求）
- **同向延伸**：同类型分型取更极值替代（顶取更高、底取更低）
- 输出：`_bi_points` 列表，每个点含 `{type, price, idx, data}`

### 3.3 MACD 面积追踪 `_bi_macd_areas`
- 每根 5m bar 累积 `|histogram|` 到 `_current_bi_macd_area`
- 每笔完成时保存面积到 `_bi_macd_areas` 列表并重置
- 用于 S6 底背驰判定（当前笔面积 vs 前同向笔面积）

### 3.4 中枢状态机 `_update_pivots()`
这是 **iter4 最核心的结构性改动**，从"滑动窗口重算"升级为状态机：

```
状态流转：
None → forming：检测到3笔重叠区间
forming → active：第4笔仍在中枢范围内（延伸）
forming/active → left_up：某笔 low > ZG（向上离开）
forming/active → left_down：某笔 high < ZD（向下离开）
left_* → None：离开后又形成新中枢（旧中枢归档）
```

- **中枢定义**：3笔重叠区间的交集 `[ZD, ZG]`（ZD=交集下沿，ZG=交集上沿）
- **3B/3S 只在 `left_up`/`left_down` 状态下生成**
- **2B/2S 只在 `forming`/`active` 状态下生成**（S2b 闸门）
- 中枢归档后保留在 `_pivots` 列表中供参考

---

## 四、信号生成 `_check_signal()`

### 4.1 3B 买点（主信号）
```python
条件：
1. 活跃中枢存在且 state == 'left_up'
2. 当前端点 p_now 是 bottom（回踩形成）
3. p_now.price > ZG（回踩不破中枢高点）
4. S4: p_now.price < ZG + max_pullback_atr × ATR（不追太高）
5. is_bull（15m MACD diff > dea）
6. R1: 同中枢入场次数 < max_pivot_entries
```

### 4.2 3S 卖点（平多信号）
```python
条件：
1. 活跃中枢存在且 state == 'left_down'
2. 当前端点 p_now 是 top（回抽形成）
3. p_now.price < ZD（回抽不破中枢低点）
4. is_bear（15m MACD diff < dea）
→ 生成 CloseLong 信号（只平仓，不开空）
```

### 4.3 2B 辅助买点
```python
条件（S2b 闸门：需活跃中枢且 state in forming/active）：
1. p_now 是 bottom
2. p_now.price > p_prev.price（低点抬高）
3. B10: _eval_div_condition(diff_ok)
   - div_mode=1(OR): diff增强 或 面积背驰
4. is_bull
5. S9: hist_gate 确认（默认禁用）
```

### 4.4 S6 底背驰补充买点
```python
条件：
1. 有活跃中枢结构
2. p_now 是 bottom
3. p_now.price <= p_prev.price（价格创新低/持平）
4. _has_area_divergence() == True（当前笔面积 < 前同向笔 × 0.50）
5. is_bull
```

### 4.5 信号执行流程
信号生成后不立即成交，而是设为 `_pending_signal`，在下一根 1m bar 的 `_check_entry_1m()` 中验证并执行：
- Buy：验证 bar.high ≥ trigger_price 才开仓
- CloseLong：验证 bar.low ≤ trigger_price 才平仓
- 冷却期内（`_cooldown_remaining > 0`）：pending_signal 被清除

---

## 五、风控系统

### 5.1 止损体系（三层）

| 层级 | 名称 | 触发条件 | 实现 |
|------|------|----------|------|
| P1 | 硬止损 | bar.low ≤ stop_price | `_check_stop_loss_1m()` |
| S5 | 早期保护 | 开仓后 ≤ min_hold_bars 根5m bar | 止损距离加倍（`effective_stop`） |
| P3 | ATR移动止损 | 浮盈 > activate_mult × ATR | high - trailing_mult × ATR |

**S5 最小持仓保护**：开仓后前2根5m bar内，止损距离翻倍。原因：短持仓(≤2bar)在统计上几乎全亏，加宽止损减少噪声出局。

**S7 笔低点结构trailing**：trailing 激活后，如果最近有高于入场价+ATR的 bottom 端点，用 `bottom.price - buffer` 作为止损（取ATR trailing 和笔低点中较高者）。

### 5.2 连亏断路器（两层）

| 层级 | 参数 | 效果 |
|------|------|------|
| L1 | cooldown_losses=2, bars=20 | 温和限制：冷却20根5m bar |
| L2 | circuit_breaker_losses=6, bars=60 | 强制暂停：冷却60根5m bar(≈5小时) |

连亏计数在每次止损或 CloseLong 平仓时更新（`_update_loss_streak(pnl)`），盈利时重置。

### 5.3 入场过滤

| 过滤器 | 参数 | 说明 |
|--------|------|------|
| ATR距离 | atr_entry_filter=2.0 | 触发价与止损距离 < 2×ATR 才放行 |
| R1去重 | dedup_bars=0(禁用) | 近期相同价格区域不重复入场 |
| S4深度 | max_pullback_atr=4.0 | 3B回踩点不能离ZG太远 |
| 中枢限次 | max_pivot_entries=2 | 同一中枢最多入场2次 |

---

## 六、迭代演进史

### 6.1 总览

| 迭代 | 3合约基线 | 7合约TOTAL | 除p2209 | 核心改动 |
|------|----------|-----------|---------|---------|
| **原始** | -869.6 | — | — | 基础缠论实现 |
| **iter1** | +676.4 | — | — | 15m空头过滤 + ATR优化 + 连亏冷却 |
| **iter2** | — | — | — | 扩展到7合约基准 + 中枢审计 |
| **iter3** | — | — | — | 只做多(不开空) + 手续费控制 |
| **iter4** | — | ~9600 | — | **中枢状态机**（最大结构改动） |
| **iter5** | — | ~9800 | — | 状态机调优 + 离开段确认 |
| **iter6** | — | 10191 | 3028 | R2断路器(cb=6/60) |
| **iter7** | — | 10607→11117 | 3868→4180 | S2b/S5/S4/S7/S6 组合 |
| **iter8** | — | 11117 | 4180 | 段背驰验证失败,保持不变 |

### 6.2 iter1: 从亏损到盈利（3合约）

**起点**: p2601 +1153 / p2405 -87 / p2209 -1936，合计 **-870 pts**

关键改动：
1. **15m MACD 空头过滤**：diff<0 时禁止做空 → p2209 从 -1936 翻正（+10689 贡献）
2. **ATR 参数优化**：activate=2.5, trailing=3.0 → 减少过早止损
3. **连亏冷却 L1**：losses=4, bars=30 → p2209 再提升 +3564

代价：p2601 从 +1153 退化到 +374（空头过滤砍掉了空头利润）

**终点**: 合计 **+676 pts**

### 6.3 iter2-3: 扩展合约集 + 只做多

- 扩展到 7 合约基准（Wind 数据全量回测）
- **关键决策：只做多**。iter1-3 期间发现做空在棕榈油上系统性亏损，3S 从"反手开空"改为"只平多"
- 建立 `run_7bench.py` 标准化回测脚本

### 6.4 iter4-5: 中枢状态机（最大结构改动）

**问题诊断**：iter1-3 审计发现中枢是"滑动窗口"式——每次新笔都重算最近3笔的交集，导致：
- 中枢不稳定，信号频繁误触发
- 无法区分"中枢内"和"已离开中枢"
- 3B/3S 在中枢内震荡时反复开仓（p2301 主要亏损来源）

**解决方案**（iter4）：实现最小状态机 `forming → active → left_up/left_down`
- 3B/3S 只在 left 状态下生成（真正离开后才触发）
- 中枢内震荡不再产生假3B/3S

这是整个策略演进中**改动半径最大、影响最深**的一次改造。

### 6.5 iter6: 分层断路器

**GPT-5.2 审计建议**：在 L1 冷却(2连亏/20bar)基础上增加 L2 强制断路(6连亏/60bar)。

效果：
- TOTAL: 9902 → 10191 (+289)
- p2301: -605 → -504（改善但仍亏损）
- 除p2209: 2825 → 3028 (+203)

### 6.6 iter7: 五项结构改动组合（除p2209 +1152）

这是收益最大的一轮迭代，通过5项改动将除p2209从3028提升到4180：

| 改动 | 思路 | 效果 |
|------|------|------|
| **S2b: 2B结构闸门** | 2B/2S只在forming/active中枢时允许 | 阻止无结构环境的趋势追随 |
| **S5: 最小持仓保护** | 开仓后2根5m bar内止损距离加倍 | 减少噪声出局，统计验证≤2bar全亏 |
| **S4: 回踩深度限制** | 3B回踩点 < ZG + 4×ATR | 过滤追高假回踩 |
| **S7: 笔低点trailing** | trailing用最近bi bottom作止损参考 | 更结构化的止损跟随 |
| **S6: 面积背驰** | 底背驰(价格新低+面积衰减)作补充买点 | div_mode=1, threshold=0.50 |

**否决的改动**：
- S3(3B两步确认): 杀p2209(-7600pts)，趋势合约不能延迟入场
- min_hold_bars=3: 杀p2209
- S2c(仅active允许2B): 过严

### 6.7 iter8: 段背驰方案全面验证失败

**目标**：把2B/2S从"diff点值比较"升级为"走势段MACD面积力度比较"

**实现**：
- `_build_segments()`: 从笔端点构建走势段（前一同类端点突破判定段切换）
- `_check_seg_divergence()`: 两个同向段的面积衰减判定

**Bug发现与修复**：
- 段构建的极值追踪用全局max/min → 818笔只生成10段
- 修复为"前一端点"追踪后正常（208段，平均3.9笔/段）

**测试矩阵（全部失败）**：

| 用法 | 结果 | 原因 |
|------|------|------|
| 替代原2B | -11400 | 反转信号≠趋势延续，杀全部入场 |
| 补充入场 | -710 | 背驰信号质量差 |
| 仅做出场 | -1933 | 过早平仓 |
| 过滤2B(力度递增) | -1635 | 杀趋势合约 |
| 过滤2B(段长度) | -1482 | 杀p2601 |
| histogram门 | -1328~-3350 | 过滤过猛 |

**结论**：段背驰在1m/5m级别噪声太大，不适用于该策略。保持iter7配置不变。

**同期发现**：策略 `on_bar()` 对非交易时段bar的止损/入场处理有bug，已修复（加session过滤）。

---

## 七、当前参数与性能

### 7.1 最终参数
```python
# MACD
macd_fast=12, macd_slow=26, macd_signal=9

# ATR
atr_window=14, atr_trailing_mult=2.0, atr_activate_mult=2.0
atr_entry_filter=2.0

# 笔/中枢
min_bi_gap=5, pivot_valid_range=6

# 风控
cooldown_losses=2, cooldown_bars=20          # L1
circuit_breaker_losses=6, circuit_breaker_bars=60  # L2
min_hold_bars=2                               # S5
max_pullback_atr=4.0                          # S4
use_bi_trailing=True                          # S7
stop_buffer_atr_pct=0.02

# 入场过滤
max_pivot_entries=2, pivot_reentry_atr=0.6
dedup_bars=0  # 禁用

# 背驰
div_mode=1, div_threshold=0.50               # S6

# 其他
fixed_volume=1
seg_enabled=False, hist_gate=0               # S8/S9 禁用
```

### 7.2 7合约回测结果

| 合约 | PnL(pts) | trades | Sharpe | 数据源 | 环境 |
|------|----------|--------|--------|--------|------|
| p2209 | +6937.6 | 116 | 4.19 | Wind | 趋势↑(极强) |
| p2601 | +1246.9 | 124 | 3.55 | XT | 趋势↑(温和) |
| p2501 | +1164.4 | 281 | 1.90 | Wind | 趋势↑(长周期) |
| p2405 | +888.5 | 139 | 2.64 | Wind | 震荡/转折 |
| p2505 | +558.0 | 124 | 1.56 | Wind | 下跌 |
| p2301 | +175.0 | 100 | 0.45 | Wind | 震荡/低波动 |
| p2509 | +146.9 | 106 | 0.80 | Wind | 温和上涨 |

**汇总**: TOTAL=**11117.3** pts | 除p2209=**4179.7** | 全部7合约为正 | 13合约盈9亏4

### 7.3 回测口径
- 框架：vnpy BacktestingEngine
- K线：1分钟（策略内部合成5m/15m）
- slippage=1.0, rate=0.0001, size=10.0, pricetick=2.0
- 初始资金：100万
- 数据归一化：`normalize_1m_bars()` 过滤非交易时段

---

## 八、代码结构

### 8.1 文件索引

| 文件 | 说明 |
|------|------|
| `src/qp/strategies/cta_chan_pivot.py` | 策略主文件（~1500行） |
| `src/qp/datafeed/normalizer.py` | 数据归一化 + session-aware 窗口计算 |
| `src/qp/backtest/engine.py` | 回测引擎封装 |
| `src/qp/utils/chan_debugger.py` | 缠论调试工具 |
| `strategies/cta_chan_pivot.py` | GUI桥接文件 |
| `scripts/run_7bench.py` | 7合约标准化回测脚本 |
| `scripts/diag_iter7.py` | 诊断脚本 |

### 8.2 关键函数清单

| 函数 | 行数 | 职责 |
|------|------|------|
| `on_bar()` | ~60 | 1m bar 入口：session过滤→止损→入场→15m/5m合成 |
| `_on_5m_bar()` | ~30 | 5m 处理链入口：包含→分型→笔→MACD/ATR→中枢→信号 |
| `_process_inclusion()` | ~40 | 包含处理 |
| `_process_bi()` | ~50 | 分型识别 + 严格笔构建 |
| `_update_pivots()` | ~80 | 中枢状态机（forming/active/left） |
| `_check_signal()` | ~120 | 信号生成（3B/3S/2B/2S/S6） |
| `_check_entry_1m()` | ~40 | 1m级入场执行 |
| `_check_stop_loss_1m()` | ~50 | 1m级止损检查（含S5早期保护） |
| `_update_trailing_stop()` | ~60 | 三阶段移动止损 |
| `_update_loss_streak()` | ~20 | 连亏计数（L1/L2断路器） |
| `_build_segments()` | ~90 | 走势段构建（S8，当前禁用） |
| `_check_seg_divergence()` | ~30 | 段背驰判定（S8，当前禁用） |

---

## 九、已知局限与未来方向

### 9.1 架构天花板
除p2209 约4180 pts，距目标5600仍差~1420。弱合约（p2301, p2509）的根本问题是行情特征（低波动震荡）不适合该策略的趋势跟随框架。

### 9.2 已验证无效的方向
- 段背驰（入场/出场/过滤）：信号质量差
- histogram 确认门：过滤过猛
- 紧trailing（activate<2.0）：杀p2601
- 全局降频（trade_interval）：伤趋势合约
- 15m diff>0 硬过滤：杀p2209

### 9.3 可能的突破方向
1. **更高级别信号**：在15m/60m层面识别买卖点，而非仅用15m做过滤
2. **多策略组合**：缠论策略覆盖趋势市，另一套策略覆盖震荡市
3. **动态参数切换**：根据ATR分位数/中枢宽度自动切换参数组
4. **更优合约选择**：替换p2301/p2509为更适合该策略的合约

---

## 十、复现指南

### 10.1 回测脚本
```bash
cd work/quant/quantPlus
.venv/Scripts/python.exe scripts/run_7bench.py
# 输出: TOTAL=11117.3 pts, STATUS=PASS
```

### 10.2 GUI 回测
1. 确保数据库有归一化数据（运行 `run_7bench.py` 或手动导入）
2. 打开 vnpy CTA回测 GUI
3. 选择 CtaChanPivotStrategy，设置合约/日期/参数
4. 回测参数：slippage=1.0, rate=0.0001, size=10.0, pricetick=2.0, capital=1000000

### 10.3 参数覆盖
```bash
# 单参数测试
.venv/Scripts/python.exe scripts/run_7bench.py circuit_breaker_losses=8

# 多参数组合
.venv/Scripts/python.exe scripts/run_7bench.py min_hold_bars=3 max_pullback_atr=3.0

# 保存结果
.venv/Scripts/python.exe scripts/run_7bench.py --output=experiments/test.json
```
