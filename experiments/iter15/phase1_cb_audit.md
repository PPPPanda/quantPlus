# Iter15 Phase 1: 连亏断路器审计

## 当前实现（简单版）

```python
# 连续亏损计数
if pnl < 0:
    consecutive_losses += 1
    if consecutive_losses >= 7:
        pause_remaining = 70  # bars
        consecutive_losses = 0
else:
    consecutive_losses = 0  # 一笔盈利就重置
```

## 已知问题

1. **噪音敏感**：一笔小盈利就重置计数，6连亏+1小盈+5连亏 = 不触发断路器
2. **幅度无感**：3笔大亏(共-500pts) vs 7笔小亏(共-100pts) 按相同逻辑处理
3. **固定恢复**：不管市场是否恢复，暂停70bar后就恢复交易
4. **单级别**：没有警告/降级机制，直接从正常→暂停

## 改进方向（文献/搜索）

### 1. 滚动窗口累计亏损断路器
- 追踪最近 N 笔的**累计 PnL** 而非连续亏损次数
- 触发条件：`sum(recent_N_trades) < -threshold`
- 优点：抗噪音（小盈利不会重置窗口）
- 参考：已有 `dd_window_trades` 参数但使用 ATR 作为阈值

### 2. 幅度加权断路器
- 设计：`weighted_loss = sum(max(0, -pnl / ATR) for pnl in recent_trades)`
- 大亏按 ATR 倍数计权，小亏权重低
- 触发条件：`weighted_loss > threshold`

### 3. 自适应恢复
- **信号驱动恢复**：等待下一个"高质量信号"（如3B+MACD背驰确认）
- **波动率恢复**：等待 ATR 回到正常范围
- **权益曲线恢复**：等待短期 MA 向上穿越长期 MA

### 4. 分级断路器
```
Level 0: 正常交易
Level 1: 警告（仓位减半，或只做3B不做2B）
Level 2: 限制（仅做高质量信号）
Level 3: 暂停（完全停止）
```
- 升级条件：连续触发下一级
- 降级条件：盈利笔数累积

### 5. Drawdown-Based 断路器
- 追踪策略净值的 rolling peak
- 当 drawdown 超过阈值时暂停
- 这是 **最本质的风控**——直接控制亏损而非间接指标
