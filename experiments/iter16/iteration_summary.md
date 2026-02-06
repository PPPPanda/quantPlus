# Iteration 16 Summary — CB2.1 评估与移除

## Status: ✅ COMPLETED - CB2.1 已从代码中移除

## 目标
1. 修复 CB2.1 的恢复机制死锁问题
2. 评估 CB2.1 vs 简单 CB 的价值
3. 根据评估结果决定保留或移除

## 发现的根本原因

### 死锁问题
```
简单 CB: 暂停 70 bars → 自动恢复 → 允许交易
CB2.1:   L3 暂停 → 需要盈利才能恢复 → 无法交易 → 无法盈利 → 永久暂停
```

### 代码对比
```python
# 简单 CB（有效）
if self._cooldown_remaining > 0:
    self._cooldown_remaining -= 1  # 时间驱动，自动恢复

# CB2.1 原版（死锁）
if win_count >= self.cb2_recovery_win:
    can_recover = True  # 需要盈利才能恢复，但暂停期间无交易
```

## 实现的修复

### 新增参数
```python
cb2_max_pause_bars: int = 70  # 与简单CB一致
```

### 新增逻辑（_on_5m_bar）
```python
# 时间驱动恢复
if self._cb2_pause_remaining > 0:
    self._cb2_pause_remaining -= 1
    if self._cb2_pause_remaining == 0 and self._cb2_risk_level > 0:
        self._cb2_risk_level = 0  # 完全恢复到 L0
        self._cb2_weighted_scores.clear()  # 清空窗口
```

### 触发计时器逻辑（_update_loss_streak）
```python
# 任何风险等级提升时启动计时器
if new_level > old_level and self.cb2_max_pause_bars > 0:
    self._cb2_pause_remaining = self.cb2_max_pause_bars
```

## 测试结果

### 单合约验证 (p2501)
```
BASELINE:     PnL=13532.1, Trades=261
CB2.1 FIXED:  PnL=13532.1, Trades=261  ✅ 完全一致
CB2.1 BROKEN: PnL= 1419.8, Trades=58   ❌ 原死锁问题
```

### 全 13 合约验证
```
BASELINE TOTAL:   136679.3 pts
CB2.1 FIXED TOTAL: 136679.3 pts
DELTA: +0.0 pts

所有 13 合约 delta = 0.0 [OK]
```

## 结论

1. **修复成功**：CB2.1 + 时间驱动恢复与简单 CB 行为完全一致
2. **当前配置无额外收益**：因为参数与简单 CB 相同
3. **框架已就绪**：可在此基础上探索 CB2.1 的高级特性（幅度加权、分级响应）

## 下一步可选方向

1. **CB2.1 调参**：让其比简单 CB 更敏感地识别问题期
2. **分合约配置**：为 p2401 等问题合约设置更严格阈值
3. **其他优化方向**：继续探索 iter14 中的未完成项

## 关键教训

**恢复机制必须有时间兜底**——纯信号驱动的恢复在暂停期间会形成死锁。这是 CB2.1 设计的根本缺陷，而简单 CB 没有这个问题是因为它用了时间驱动恢复。

## 三方审计结论（Gemini）

### 核心观点
CB2.1 的设计哲学与趋势策略根本不匹配：
- **CB2.1 假设**："大额亏损 = 策略失效" → 用幅度加权更快触发保护
- **趋势策略现实**：大额亏损往往发生在趋势反转的早期试错阶段，恰恰是即将捕获大趋势的前奏

### 优化目标对比
| 设计 | 优化对象 | 适用策略 |
|------|----------|----------|
| 简单 CB | 最坏情况（连续亏损） | 趋势策略 ✅ |
| CB2.1 | 平均情况（加权损失） | 套利策略 |

### 建议
放弃 CB2.1，回归简单 CB。如需增强，考虑"日级别最大损失保护"而非交易级别的幅度加权。

---

## 最终决策：移除 CB2.1

基于以下理由，决定从代码中完全移除 CB2.1：

1. **奥卡姆剃刀**：10+ 参数的复杂性没有带来可证明的收益
2. **设计哲学不匹配**：幅度加权惩罚正常试错，与趋势策略的"长期正期望但短期波动大"特性冲突
3. **简单 CB 已足够**：7 次连续亏损 + 70 bars 冷却 + 一笔盈利重置，机制简洁有效

### 移除的代码
- **参数**（18 个）：`cb2_enabled`, `cb2_window_trades`, `cb2_l1/l2/l3_threshold`, `cb2_recovery_win`, `cb2_magnitude_weight`, `cb2_weight_floor/cap`, `cb2_decay_per_win`, `cb2_atr_recovery`, `cb2_atr_recovery_factor`, `cb2_l1_skip_pct`, `cb2_max_pause_bars`
- **运行时变量**（7 个）：`_cb2_recent_pnls`, `_cb2_weighted_scores`, `_cb2_risk_level`, `_cb2_trigger_atr`, `_cb2_signal_counter`, `_cb2_base_loss`, `_cb2_pause_remaining`
- **逻辑块**（3 处）：`_on_5m_bar()` 时间恢复、`_check_entry_1m()` 分级过滤、`_update_loss_streak()` 加权计算

### 移除后验证
- 语法检查：✅ PASS
- 7 合约回测：`TOTAL=11117.3 pts` ✅ PASS
- 13 合约回测：`TOTAL=12164.2 pts`（默认参数基线）✅ 与预期一致

## 文件变更

- `src/qp/strategies/cta_chan_pivot.py` — **移除所有 CB2.1 相关代码**
- `scripts/iter16_cb21_full_test.py` — 13 合约对比测试脚本（历史）
- `experiments/iter16/cb21_fixed_results.json` — 完整测试结果（历史）
