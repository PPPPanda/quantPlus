# iter10 Phase 1: 止盈/锁盈策略审计

## 当前出场机制分析

### 1. 硬止损 (P1)
- 开仓时设定: stop = 回踩底点 - buffer
- buffer = max(pricetick, atr * 0.02)
- 固定不动，直到 trailing 接管

### 2. 最小持仓保护 (S5)
- 前 2 根 5m bar: 止损距离加倍（防噪声打止损）

### 3. 锁盈 (R3 Phase 2) — 当前禁用
- `lock_profit_atr=0.0` → 完全禁用
- 设计: 浮盈 ≥ 1R 时，止损抬至 entry + lock_profit_atr × ATR
- 这是一个 break-even 止损机制

### 4. Trailing Stop (R3 Phase 3)
- 激活条件: 浮盈 > atr_activate_mult(2.5) × ATR
- 追踪: high - atr_trailing_mult(3.0) × ATR
- 加上 bi_trailing: 用最近笔底点作为止损参考

### 5. 缠论信号平仓 (3S/2S)
- 向下离开中枢 + 回抽不破 ZD → 平多
- 顶背驰 (2S) → 平多

## 问题识别

### 从 entry 到 trailing 激活之间有"无保护区"
- 当 ATR=40 时, 激活需要浮盈 > 100 (2.5×40)
- 在这之前, 止损始终是初始硬止损
- 如果价格先涨 80 再跌回来, 利润全吐光

### lock_profit_atr 正好可以填补这个空隙
- 浮盈 ≥ 1R 时保本 (lock_profit_atr=0)
- 或者抬到入场价以上 (lock_profit_atr=0.5)

## 测试计划
1. lock_profit_atr=0.5 → 浮盈≥1R时锁盈到 entry+0.5*ATR
2. lock_profit_atr=1.0 → 浮盈≥1R时锁盈到 entry+1.0*ATR  
3. lock_profit_atr=0.3 → 更保守的锁盈

基线: iter9 的 cb=7/70 (TOTAL=11551.1)
