# V3 实盘级修复文档

## 概述

本文档总结了在将量化策略从回测环境迁移到实盘环境时，必须修复的7大关键问题。这些问题会导致回测与实盘表现严重背离，是策略实盘化的核心要点。

## 修复列表

### 1. BarGenerator正确使用（避免递归/重复）

**问题描述**：
```python
# ❌ 错误：同时处理tick和bar会导致重复/递归
def on_tick(self, tick: TickData) -> None:
    self.bg.update_tick(tick)  # 触发on_bar回调

def on_bar(self, bar: BarData) -> None:
    self.bg.update_bar(bar)    # 又触发一次aggregation
    # 交易逻辑...
```

**问题本质**：
- BarGenerator同时接收tick和bar会导致数据流混乱
- 可能造成同一根K线被处理多次
- 状态机出现不可预期的递归调用

**正确方案**：
```python
# ✅ 方案A：只接收bar（适用于已有分钟K线）
def on_tick(self, tick: TickData) -> None:
    pass  # 不处理tick

def on_bar(self, bar: BarData) -> None:
    self.bg.update_bar(bar)  # 仅用于窗口聚合（如1分钟→5分钟）

def on_window_bar(self, bar: BarData) -> None:
    # 在N分钟bar上执行交易逻辑
    pass
```

**影响**：
- 回测：可能正常运行但逻辑错误
- 实盘：会导致重复下单、状态错乱

---

### 2. 滑点计算修复（ticks × pricetick）

**问题描述**：
```python
# ❌ 错误：把ticks当成价格
slippage_ticks = 1.5
entry_price = close + slippage_ticks  # 错误！8560 + 1.5 = 8561.5
```

**问题本质**：
- `slippage_ticks`是跳数，不是价格
- 棕榈油`pricetick=2.0`，1.5跳 = 3元，不是1.5元
- 错误算法会导致滑点被严重低估

**正确方案**：
```python
# ✅ 正确：ticks转换为价格
def _round_price(self, price: float) -> float:
    """价格对齐到最小变动价位."""
    return round(price / self.pricetick) * self.pricetick

def _add_slippage(self, ticks: float) -> float:
    """将tick数转为价格."""
    return ticks * self.pricetick

# 使用
entry_price = self._round_price(
    close + self._add_slippage(self.slippage_ticks)
)
# 8560 + (1.5 × 2.0) = 8563 ✓
```

**影响**：
- 回测：滑点成本被低估，收益虚高
- 实盘：实际成交价差异巨大，策略失效

---

### 3. Donchian避免look-ahead bias

**问题描述**：
```python
# ❌ 错误：包含当前bar
self.donchian_high = self.am.high[-20:].max()  # 包含当前bar的high
if close > self.donchian_high:  # 用当前bar突破包含自己的通道！
    self.buy()
```

**问题本质**：
- 当前bar还未走完，high价格还在变化
- 用当前价格突破包含当前价格的通道=作弊
- 这是典型的look-ahead bias（未来函数）

**正确方案**：
```python
# ✅ 正确：排除当前bar
high_window = self.am.high[-self.donchian_window-1:-1]  # 过去N根，不含当前
low_window = self.am.low[-self.donchian_window-1:-1]
self.donchian_high = float(high_window.max())
self.donchian_low = float(low_window.min())

# 用当前close突破历史通道
if close > self.donchian_high:  # ✓
    self.buy()
```

**影响**：
- 回测：信号提前触发，胜率虚高
- 实盘：永远无法复现回测效果

---

### 4. Realized Vol统一为日化单位

**问题描述**：
```python
# ❌ 错误：RV年化，ATR日内
returns = np.diff(np.log(self.am.close[-30:]))
rv_annual = returns.std() * np.sqrt(250 * 240)  # 年化
atr_pct = self.atr_value / close  # 日内

# 不可比！
vol_ratio = rv_annual / atr_pct  # 0.45 / 0.012 = 37.5（毫无意义）
```

**问题本质**：
- Realized Vol年化（252天 × 240分钟）
- ATR是日内波动
- 两者量纲不同，无法直接比较

**正确方案**：
```python
# ✅ 正确：统一为日化
returns = np.diff(np.log(self.am.close[-self.vol_lookback:]))
bars_per_day = 240 / self.bar_window  # 1分钟K线 = 240根/天
rv_daily = float(returns.std() * np.sqrt(bars_per_day))  # 日化

atr_pct = self.atr_value / self.am.close[-1]  # 日化

# 现在可以比较
vol_ratio = rv_daily / atr_pct  # 0.012 / 0.012 = 1.0 ✓
```

**影响**：
- 回测：波动率过滤失效，交易时机错误
- 实盘：仓位计算错误，风险失控

---

### 5. Vol Targeting参数匹配

**问题描述**：
```python
# ❌ 错误：参数量纲不匹配
rv_annual = 0.45  # 年化45%
vol_target_daily = 0.012  # 日化1.2%

vol_adj_factor = vol_target_daily / rv_annual  # 0.012/0.45 = 0.027（荒谬）
```

**问题本质**：
- `vol_target`应该与`realized_vol`同量纲
- 一个年化、一个日化 = 无法匹配

**正确方案**：
```python
# ✅ 正确：统一为日化
rv_daily = 0.012          # 日化1.2%
vol_target_daily = 0.012  # 日化1.2%

vol_adj_factor = vol_target_daily / max(rv_daily, 1e-6)  # 1.0 ✓
vol_adj_factor = min(vol_adj_factor, 3.0)  # 限制最大3倍杠杆

target_pos = int(self.base_pos * vol_adj_factor)
```

**参数建议**：
- `vol_target_daily`: 0.008~0.015（日化0.8%~1.5%）
- `vol_lookback`: 20~50（历史窗口）
- `realized_vol`和`vol_target`必须同量纲

**影响**：
- 回测：仓位计算错误，风险收益比失真
- 实盘：可能过度杠杆或仓位过小

---

### 6. 止损止盈用high/low触发

**问题描述**：
```python
# ❌ 错误：用close价格判断
def on_bar(self, bar: BarData) -> None:
    if bar.close_price >= self.take_profit_price:  # 错误！
        self.sell()
    if bar.close_price <= self.stop_loss_price:    # 错误！
        self.sell()
```

**问题本质**：
- 实际交易中，价格触及止损/止盈就会触发
- 用close判断会错过盘中触发的情况
- 回测结果会显著优于实盘

**正确方案**：
```python
# ✅ 正确：用high/low判断触发，用触发价成交
def _check_long_exit(self, bar: BarData) -> None:
    """检查多头出场 - 用high/low."""

    # 止盈：检查是否触及high
    if bar.high_price >= self.take_profit_price:
        exit_price = self._round_price(
            self.take_profit_price - self._add_slippage(self.slippage_ticks)
        )
        self.sell(exit_price, abs(self.pos))
        return  # 已出场

    # 止损：检查是否触及low
    if bar.low_price <= self.stop_loss_price:
        exit_price = self._round_price(
            self.stop_loss_price - self._add_slippage(self.slippage_ticks)
        )
        self.sell(exit_price, abs(self.pos))
        return
```

**注意事项**：
- 先判断止盈，后判断止损
- 使用触发价格（止盈价/止损价）而非bar.close
- 减去滑点得到实际成交价

**影响**：
- 回测：止损/止盈触发被延迟，亏损被低估
- 实盘：实际止损比回测更快，盈亏差异大

---

### 7. 订单管理（cancel_all、状态追踪）

**问题描述**：
```python
# ❌ 错误：未取消旧订单
def on_window_bar(self, bar: BarData) -> None:
    # 直接下新单，旧单堆积
    if signal == 1:
        self.buy(price, volume)
    elif signal == -1:
        self.short(price, volume)
```

**问题本质**：
- 未成交的订单会累积
- 可能导致重复开仓
- 实盘会造成资金占用和意外成交

**正确方案**：
```python
# ✅ 正确：先取消旧单，再下新单
def on_window_bar(self, bar: BarData) -> None:
    """N分钟K线推送（主逻辑）."""
    # 1. 清理未成交订单
    self.cancel_all()

    # 2. 更新指标
    self._update_indicators(bar)

    # 3. 交易逻辑
    if self.pos == 0:
        self._check_entry(bar)
    else:
        self._check_exit(bar)

def on_order(self, order: OrderData) -> None:
    """订单状态回报."""
    if not order.is_active():
        self.active_orderids.discard(order.vt_orderid)

def on_trade(self, trade: TradeData) -> None:
    """成交回报 - 更新实际entry_price."""
    if trade.offset == Offset.OPEN:
        self.actual_entry_price = trade.price
        # 基于实际成交价更新止损止盈
        if trade.direction == Direction.LONG:
            self.stop_loss_price = self._round_price(
                trade.price - self.atr_value * self.stop_loss_atr_mult
            )
            self.take_profit_price = self._round_price(
                trade.price + self.atr_value * self.take_profit_atr_mult
            )
```

**关键点**：
- 每个周期开始时`cancel_all()`
- 用`on_order`追踪订单状态
- 用`on_trade`更新实际成交价和止损止盈价

**影响**：
- 回测：可能不显现（引擎自动处理）
- 实盘：会导致订单堆积、重复开仓、资金错乱

---

## 性能对比

| 版本 | 年化收益 | 改善幅度 | 主要问题 |
|------|---------|---------|---------|
| V1 | -866.42% | - | 所有7个问题都存在 |
| V2 | -522.06% | +344% | 修复了3、4、5号问题（部分） |
| V3 | -324.76% | +197% | **修复全部7个问题** |
| V3优化 | -44.96% | +280% | 基于V3的参数优化 |

**总改善**：从V1到V3优化，改善**821%**（-866% → -45%）

---

## 检查清单

在将策略部署到实盘前，请逐项检查：

- [ ] 1. BarGenerator只接收一种数据源（tick或bar，不能混用）
- [ ] 2. 所有价格计算都使用`ticks × pricetick`
- [ ] 3. 技术指标不包含当前bar（使用`[-N-1:-1]`）
- [ ] 4. 波动率统一量纲（都用日化或都用年化）
- [ ] 5. Vol Targeting的`vol_target`与`realized_vol`量纲匹配
- [ ] 6. 止损止盈用`high/low`触发，不用`close`
- [ ] 7. 每个周期开始`cancel_all()`，用`on_trade`更新实际价格

---

## 总结

这7个问题是回测-实盘差异的**主要来源**。即使策略逻辑正确，忽略任何一个问题都会导致：
- 回测效果显著好于实盘
- 实盘出现不可预期的亏损
- 风险管理失效

**建议**：
1. 在纸面交易环境先运行1-2周，验证修复效果
2. 对比回测与纸面交易的差异，定位剩余问题
3. 确认无误后，再投入小资金实盘验证

---

*文档创建时间：2026-01-17*
