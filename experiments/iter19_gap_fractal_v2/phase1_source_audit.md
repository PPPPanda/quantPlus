# Phase 1: 主控源码审计 - 节假日跳空对缠论分型的影响

## 审计时间
2026-02-05 19:55 GMT+8

## 1. 当前 S26b 实现分析

### 1.1 代码位置
`src/qp/strategies/cta_chan_pivot.py`, lines 546-565

### 1.2 当前逻辑
```python
# S26b 分级冷却
if gap_atr >= self.gap_extreme_atr:  # 1.5x ATR
    if gap_atr < self.gap_tier1_atr:  # <10x
        cooldown = 6  # 中等跳空最危险
    elif gap_atr < self.gap_tier2_atr:  # <30x
        cooldown = 3  # 大跳空次之
    else:  # >30x
        cooldown = 0  # 极大跳空反而安全
    if cooldown > 0:
        self._gap_cooldown_remaining = cooldown
```

### 1.3 冷却的实际作用（line 1055）
```python
if self._gap_cooldown_remaining > 0:
    return  # 跳过信号检查
```

### 1.4 ⚠️ 关键问题：冷却只跳过信号，不影响结构构建

| 处理阶段 | 函数 | 是否受冷却影响 |
|----------|------|----------------|
| 包含处理 | `_process_inclusion()` | ❌ 不受影响 |
| 分型识别 | `_process_bi()` | ❌ 不受影响 |
| 中枢更新 | `_update_pivots()` | ❌ 不受影响 |
| 信号检查 | `_check_signal()` | ✅ 被跳过 |

**结论**：跳空K线仍然参与分型/笔/中枢的构建，只是暂时不生成信号。冷却结束后，这些被污染的结构继续参与后续判断。

## 2. 分型处理中的跳空问题

### 2.1 `_process_inclusion()` 分析

包含处理依赖相邻K线的高低点比较：
```python
in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']
```

**跳空场景问题**：
- 跳空高开：`new_bar['low'] > last['high']` → 无包含关系，但创造价格断层
- 跳空低开：`new_bar['high'] < last['low']` → 无包含关系，但创造价格断层
- 两根K线之间的"空白区域"无法被包含处理正确处理

### 2.2 `_process_bi()` 分析

分型识别依赖三K线模式：
```python
is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']
```

**跳空场景问题**：
1. **跳空低开创造伪底分型**
   - 场景：`[H1=100, L1=95] → GAP → [H2=90, L2=85]`
   - 如果 H2 < L1，这根跳空K线的 low 可能被错误识别为底分型

2. **跳空高开创造伪顶分型**
   - 场景：`[H1=100, L1=95] → GAP → [H2=115, L2=110]`
   - 如果 L2 > H1，这根跳空K线的 high 可能被错误识别为顶分型

3. **中等跳空最危险的原因**
   - 大跳空(>30x ATR)：形成明确的价格断层，后续K线与前序结构无交集，自然隔离
   - 中等跳空(3-10x ATR)：跳空幅度不够大，跳空K线与前序K线仍有"模糊重叠"的可能
   - 这种模糊重叠导致包含处理方向混乱，分型判断失真

## 3. 笔构建中的跳空问题

### 3.1 同向延伸规则
```python
if last['type'] == 'top' and cand['price'] > last['price']:
    self._bi_points[-1] = cand  # 顶更高，延伸
elif last['type'] == 'bottom' and cand['price'] < last['price']:
    self._bi_points[-1] = cand  # 底更低，延伸
```

**跳空场景问题**：
- 跳空创造的伪分型可能导致错误的同向延伸
- 例：本应是趋势延续，但跳空创造的伪底导致"假反转"

### 3.2 严格笔间隔规则
```python
if cand['idx'] - last['idx'] >= self.min_bi_gap:  # 默认4
    self._bi_points.append(cand)  # 成笔
```

**跳空场景问题**：
- 跳空前后的K线数量减少（包含处理合并）
- 可能导致 `min_bi_gap` 条件更难满足
- 或者反过来：跳空造成的伪分型使间隔计算异常

## 4. 中枢识别中的跳空问题

### 4.1 中枢定义（3笔重叠区间）
```python
zg = min(r1[1], r2[1], r3[1])  # 三笔高点的最小值
zd = max(r1[0], r2[0], r3[0])  # 三笔低点的最大值
if zg > zd:  # 有重叠 → 形成中枢
```

**跳空场景问题**：
- 如果跳空导致的伪笔参与中枢计算
- ZG/ZD 可能被错误拉伸或压缩
- 导致：
  - 信号触发位置错误（3B/3S 基于 ZG/ZD）
  - 中枢有效范围异常

## 5. 建议的改进方向

### 5.1 低改动半径（调参优化）

| 方案 | 改动 | 原理 |
|------|------|------|
| A1 | gap_extreme_atr: 1.5→1.0 | 更早触发检测 |
| A2 | gap_cooldown_bars: 6→10 | 更长冷却期 |
| A3 | gap_tier1_atr: 10→8 | 调低中等跳空上界 |

### 5.2 中改动半径（结构层面）

| 方案 | 改动 | 原理 |
|------|------|------|
| B1 | 跳空后重置包含方向 | `_inclusion_dir = 0` |
| B2 | 跳空K线不参与分型判断 | 跳过 `_process_bi()` |
| B3 | 跳空后分型需二次确认 | 等待后续K线验证 |

### 5.3 高改动半径（结构重建）

| 方案 | 改动 | 原理 |
|------|------|------|
| C1 | 跳空后清空最近N根K线 | 完全隔离旧结构 |
| C2 | 跳空时强制结束当前笔 | 避免伪笔延伸 |
| C3 | 跳空中枢标记为"可疑" | 降低其信号权重 |

## 6. 优先级建议

1. **首选 B1（跳空后重置包含方向）**
   - 改动点单一，可追溯
   - 符合缠论"方向未定时不强制"的精神
   - 预期：减少伪分型产生

2. **次选 B2（跳空K线不参与分型）**
   - 更激进，但效果直接
   - 风险：可能错过真实的极值点

3. **保守方案 A2（延长冷却期）**
   - 不改变结构处理逻辑
   - 只是延长"观望"时间，等待结构稳定
