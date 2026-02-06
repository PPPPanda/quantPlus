# Phase 3: Bridge Bar 实现 Backlog

## 改动清单（按优先级）

### B1: Bridge Bar 插入（数据预处理层）
**失败模式** → 跳空导致结构断层
**目标** → 让跳空成为结构的一部分
**方案（缠论依据）** → 把跳空建模为最低级别走势连接（108课：缺口是连接中枢的最低级别）
**代码改动点** → `_on_5m_bar()` 中检测跳空，在处理真实bar前先插入bridge bar
**预期影响** → 分型识别更稳定，包含方向更准确
**风险** → 可能增加笔数量（bridge bar 参与分型）

### B2: 真实Bar计数（笔构建层）
**失败模式** → Bridge bar 让笔"看起来更长"，破坏 min_bi_gap 约束
**目标** → 笔最小长度按真实bar计算
**方案** → K线增加 `is_bridge` 标记，笔计数时跳过 bridge
**代码改动点** → `_process_bi()` 中用 `real_count` 替代 `idx` 距离
**预期影响** → 保持笔约束的语义一致性
**风险** → 无

### B3: 跳空力度特征（信号层）
**失败模式** → 跳空进/出中枢时信号有效性不同
**目标** → 把"是否通过跳空离开"作为信号置信度因子
**方案** → 中枢增加 `leave_by_gap` 标记，可用于后续过滤或置信度调整
**代码改动点** → `_update_pivots()` 中检测离开方式
**预期影响** → 为后续优化提供特征
**风险** → 无直接影响（仅记录）

---

## 实现计划

### Round 1: 基础实现
1. 添加参数 `bridge_bar_enabled: bool = True`
2. 在 `_on_5m_bar()` 中实现跳空检测和 bridge bar 插入
3. K线结构增加 `is_bridge` 字段
4. `_process_bi()` 中使用 `real_count`
5. 回测验证

### Round 2: 中枢力度特征
1. 中枢增加 `leave_by_gap` 标记
2. 信号过滤（可选）：gap离开时调整入场条件

### Round 3: 参数调优
1. 调整 `gap_threshold_atr`：多大的跳空才插 bridge
2. 调整 bridge bar 参与分型的规则

---

## 关键代码片段

### Bridge Bar 插入
```python
def _on_5m_bar(self, bar: dict) -> None:
    # 检测跳空
    if self._prev_close_5m > 0 and self.atr > 0 and self.bridge_bar_enabled:
        gap = bar['open'] - self._prev_close_5m
        gap_abs = abs(gap)
        
        if gap_abs > self.gap_bridge_threshold * self.atr:
            # 插入 bridge bar
            bridge = {
                'datetime': bar['datetime'],
                'open': self._prev_close_5m,
                'high': max(self._prev_close_5m, bar['open']),
                'low': min(self._prev_close_5m, bar['open']),
                'close': bar['open'],
                'diff': self.diff_5m,  # 继承当前指标
                'atr': self.atr,
                'diff_15m': self._prev_diff_15m,
                'dea_15m': self._prev_dea_15m,
                'is_bridge': True,
            }
            self._process_bar_data(bridge)
    
    # 处理真实 bar
    bar_data = {**bar, 'is_bridge': False}
    self._process_bar_data(bar_data)
```

### 真实Bar计数
```python
def _count_real_bars(self, start_idx: int, end_idx: int) -> int:
    """计算两个索引之间的真实bar数量（排除bridge）"""
    count = 0
    for i in range(start_idx, min(end_idx + 1, len(self._k_lines))):
        if not self._k_lines[i].get('is_bridge', False):
            count += 1
    return count
```
