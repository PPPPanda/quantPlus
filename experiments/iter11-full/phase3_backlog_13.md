# Phase3: 改动清单（Backlog）

## 已完成
- [x] cb=7/70 验证（iter9-full）
- [x] div_threshold 0.35/0.39/0.45/0.50 扫描
- [x] max_pullback_atr 2.5/3.0/3.5/4.0 扫描
- [x] atr_activate_mult 2.0/2.5 对比
- [x] atr_trailing_mult 3.0/3.5 对比
- [x] cooldown_bars 15/20/30 对比
- [x] atr_entry_filter 2.0/2.5 对比
- [x] hist_gate 0/1 对比
- [x] circuit_breaker 5~8 扫描

## 最佳参数（推荐应用）
```
circuit_breaker_losses=7, circuit_breaker_bars=70  (was 6/60)
div_threshold=0.39                                  (was 0.50)
max_pullback_atr=3.5                                (was 4.0)
atr_activate_mult=2.0                               (was 2.5)
```
其余保持默认。

## 待研究（代码级改动方向）

### P1: p2401 死亡共振对策
- [ ] 实现"death spiral detector"：连续N根bar或N天亏损后暂停该合约
- [ ] 波动率regime检测：低波动+下跌环境禁止做多
- [ ] 合约级参数override机制（允许对特定合约使用不同参数）

### P2: 止盈研究
- [ ] 时间止盈：持仓超N根bar后收紧trailing（上轮结论：lock_profit_atr全系列破坏弱合约）
- [ ] 分批减仓：达到xATR利润后减仓50%
- [ ] 目标价位止盈：基于中枢+笔结构设定目标

### P3: 节假日保护
- [ ] 13合约视角重新验证（7合约结论：overnight利润占62%，保护不可行）
- [ ] 仅分析4个负合约的节假日跳水损失占比

### P4: 做空对称性
- [ ] 分析代码中做多做空逻辑对称性
- [ ] 评估是否有做多偏向bug
- [ ] 但棕榈油做空灾难性亏损是已知限制

## 不推荐的方向
- ❌ hist_gate > 0（毁策略）
- ❌ min_hold_bars > 2（数据异常）
- ❌ atr_trailing_mult > 3.0（灾难性回撤）
- ❌ circuit_breaker > 7 或 < 7
- ❌ 全局降频/diff>0硬过滤/紧trailing（历史教训）
