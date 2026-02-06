# Phase 1 主控审计：跳空对缠论代码实现的影响

**审计角度**：代码实现层面
**审计日期**：2025-02-05

---

## 1. 当前代码中跳空相关处理现状

### 1.1 跳空检测：❌ 无
```python
# 当前代码中没有任何跳空检测逻辑
# grep 搜索 "gap|跳空|extreme|skip" 无结果
```

### 1.2 相关保护机制

| 机制 | 实现状态 | 对跳空的保护作用 |
|------|---------|-----------------|
| 连亏断路器 (CB) | ✅ 已实现 | 事后保护，需先亏 7 次才触发，对首次跳空无效 |
| ATR 止损 | ✅ 已实现 | 滞后，使用节前 ATR，第一根 bar 保护不足 |
| S26 极端跳空安全网 | ❌ 未实现 | iter17 推荐但未落地 |
| S27 ATR 自适应 | ❌ 未实现 | iter17 推荐但未落地 |

---

## 2. 跳空对各缠论结构的代码层影响

### 2.1 分型识别 (`_process_bi()`)

**当前实现**：
```python
is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']
```

**跳空影响**：
- 跳空 K 线通常幅度异常（日均波动 44%）
- 容易形成"假分型"：
  - 向上跳空 + 高点回落 → 假顶分型
  - 向下跳空 + 低点反弹 → 假底分型
- **代码无法区分正常分型和跳空导致的假分型**

**风险等级**：🟡 中

### 2.2 包含处理 (`_process_inclusion()`)

**当前实现**：
```python
if self._inclusion_dir == 1:  # 向上包含
    merged['high'] = max(last['high'], new_bar['high'])
    merged['low'] = max(last['low'], new_bar['low'])
else:  # 向下包含
    merged['high'] = min(last['high'], new_bar['high'])
    merged['low'] = min(last['low'], new_bar['low'])
```

**跳空影响**：
- 跳空后第一根 K 线与前一日收盘 K 线之间**没有包含关系**（价格区间不重叠）
- 但跳空 K 线幅度异常大，可能导致后续多根 K 线被它"吞没"
- **包含方向 `_inclusion_dir` 可能被跳空 K 线错误设定**

**风险等级**：🟡 中

### 2.3 严格笔构建 (`_process_bi()`)

**当前实现**：
```python
# 异向成笔（严格笔要求间隔 >= min_bi_gap）
if cand['idx'] - last['idx'] >= self.min_bi_gap:
    self._bi_points.append(cand)
```

**跳空影响**：
- 跳空 K 线可能被错误标定为笔端点
- 假分型直接导致假笔
- `min_bi_gap=4` 的约束无法识别跳空导致的结构失真
- **笔的方向和长度可能严重失真**

**风险等级**：🔴 高

### 2.4 中枢识别 (`_update_pivots()`)

**当前实现**：
```python
zg = min(r1[1], r2[1], r3[1])  # 三笔高点的最小值
zd = max(r1[0], r2[0], r3[0])  # 三笔低点的最大值
if zg > zd:  # 有重叠区间
    new_pivot = {...}
```

**跳空影响**：
- 跳空可能让价格直接"穿越"中枢区间
- 中枢的 ZG/ZD 可能被跳空 K 线的极端价格扭曲
- 状态机 `left_up`/`left_down` 可能被错误触发
- **产生"假突破"或"假跌破"信号**

**风险等级**：🔴 高

### 2.5 MACD 背驰 (`_update_macd_5m()`, `_has_area_divergence()`)

**当前实现**：
```python
# 增量 EMA 计算
self._ema_fast_5m = alpha_fast * close + (1 - alpha_fast) * self._ema_fast_5m
self._ema_slow_5m = alpha_slow * close + (1 - alpha_slow) * self._ema_slow_5m
diff = self._ema_fast_5m - self._ema_slow_5m
```

**跳空影响**：
- 跳空导致 close 突变，MACD 产生异常读数
- histogram 突然放大/缩小
- **面积背驰计算失真**，可能产生错误的背驰信号
- EMA 需要多根 bar 才能消化跳空冲击

**风险等级**：🔴 高

---

## 3. 需要实现的保护机制

### 3.1 S26 极端跳空安全网（推荐 ✅）

**设计**：
```python
# 参数
extreme_gap_atr: float = 3.0      # 触发阈值（×ATR）
gap_skip_bars: int = 3            # 跳过 bar 数（15分钟）

# 检测逻辑（在 _on_5m_bar 开头）
if self._prev_close_5m > 0:
    gap = abs(bar['open'] - self._prev_close_5m)
    if gap > self.extreme_gap_atr * self.atr:
        self._gap_skip_remaining = self.gap_skip_bars
        
# 信号生成时检查
if self._gap_skip_remaining > 0:
    self._gap_skip_remaining -= 1
    return  # 跳过信号检查
```

**预期效果**：
- 仅在极端跳空（>3×ATR，约 150+ pts）时触发
- 暂停前 15 分钟的信号生成，等待结构稳定
- 不影响普通交易日

### 3.2 S27 ATR 自适应（推荐 ✅）

**设计**：
```python
# 参数
gap_atr_threshold: float = 1.5    # 触发 ATR 调整的跳空阈值
atr_boost_factor: float = 1.5     # ATR 放大系数
atr_boost_bars: int = 6           # 放大持续 bar 数（30分钟）

# 检测逻辑
if gap > self.gap_atr_threshold * self.atr:
    self._atr_boost_remaining = self.atr_boost_bars
    
# ATR 使用时检查
def _get_effective_atr(self) -> float:
    if self._atr_boost_remaining > 0:
        return self.atr * self.atr_boost_factor
    return self.atr
```

**预期效果**：
- 跳空后临时放大 ATR 基准
- 止损更宽松，适应跳空波动
- 不阻止信号生成，只调整风控

### 3.3 跳空分型过滤（新提议）

**设计**：
```python
# 在分型识别后添加检查
def _is_gap_distorted_fractal(self, bar: dict) -> bool:
    """检查分型是否被跳空扭曲."""
    if self._prev_close_5m == 0:
        return False
    gap = abs(bar['open'] - self._prev_close_5m)
    bar_range = bar['high'] - bar['low']
    # 如果跳空 > 50% 的 K 线幅度，认为分型可能失真
    return gap > bar_range * 0.5 and gap > self.atr * 1.0
```

**预期效果**：
- 识别跳空导致的假分型
- 延迟分型确认（等待 1-2 根 bar 验证）

---

## 4. 审计结论

### 4.1 核心发现
1. **当前代码对跳空完全没有处理**，存在重大结构性风险
2. **笔和中枢是主要受影响点**，可能产生错误信号
3. **iter17 推荐的 S26/S27 未落地**，需要本次迭代实现

### 4.2 优先级排序
| 优先级 | 方案 | 实现复杂度 | 预期收益 |
|--------|------|-----------|---------|
| P0 | S26 极端跳空安全网 | ⭐ 低 | 高（防止极端亏损） |
| P1 | S27 ATR 自适应 | ⭐ 低 | 中（减少单笔亏损） |
| P2 | 跳空分型过滤 | ⭐⭐ 中 | 中（提高信号质量） |
| P3 | 结构重置 | ⭐⭐⭐ 高 | 低（风险大于收益） |

### 4.3 待三方汇总后决策
- 等待 GPT-5.2 缠论层面审计
- 等待 Gemini 交叉验证
- 汇总后确定最终实施方案
