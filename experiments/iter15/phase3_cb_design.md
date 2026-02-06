# Iter15 Phase 3: 增强版断路器设计

## 三方审核综合

### Claude（搜索）
- Daily loss limits + Consecutive loss counters + Volatility circuit breakers
- Drawdown-based 是最本质的风控

### Gemini
- 滚动窗口（20笔）累计回撤
- 幅度加权：`weighted_score = sum(|loss| * (|loss|/avg_loss))`
- 三级：黄(3%)-橙(6%)-红(10%)
- 自适应恢复：检测新中枢形成后恢复

### GPT（待补充）

## 增强版断路器设计（CB2.0）

### 核心改进

1. **滚动窗口替代连续计数**
   - 追踪最近 N 笔的累计 PnL
   - 小盈利不再重置窗口
   - 参数：`cb2_window_trades = 10`

2. **ATR归一化亏损**
   - 用 ATR 归一化亏损幅度，使不同合约可比
   - `normalized_loss = -pnl / ATR`
   - 大亏（>2*ATR）权重更高

3. **三级断路器**
   - **L1 警告**（cum_loss > 4*ATR）：仅允许3B信号，禁止2B
   - **L2 限制**（cum_loss > 6*ATR）：仓位减半（暂不实现）
   - **L3 熔断**（cum_loss > 8*ATR）：暂停交易

4. **自适应恢复**
   - 不用固定 bar 数
   - 恢复条件：窗口内出现盈利 OR ATR 恢复正常

### 伪代码

```python
class EnhancedCircuitBreaker:
    def __init__(self):
        self.window_size = 10
        self.recent_pnls = deque(maxlen=10)
        self.risk_level = 0  # 0=normal, 1=warning, 2=restricted, 3=paused
        
    def update(self, pnl: float, atr: float):
        # 归一化亏损
        if atr > 0:
            norm_pnl = pnl / atr
        else:
            norm_pnl = pnl / 100  # fallback
        
        self.recent_pnls.append(norm_pnl)
        
        # 计算窗口内累计
        window_sum = sum(self.recent_pnls)
        
        # 分级判断
        if window_sum < -8:
            self.risk_level = 3  # 熔断
        elif window_sum < -6:
            self.risk_level = 2  # 限制
        elif window_sum < -4:
            self.risk_level = 1  # 警告
        else:
            self.risk_level = 0  # 正常
            
    def is_signal_allowed(self, signal_type: str) -> bool:
        if self.risk_level == 3:
            return False
        if self.risk_level == 2:
            return signal_type == '3B'  # 仅3B
        if self.risk_level == 1:
            return signal_type in ['3B', '2B']  # 禁止低质量信号
        return True
        
    def should_reduce_position(self) -> bool:
        return self.risk_level >= 2
```

### 参数建议
- `cb2_window_trades = 10`（约2-3天交易量）
- `cb2_l1_threshold = -4`（ATR归一化）
- `cb2_l2_threshold = -6`
- `cb2_l3_threshold = -8`

### 与现有参数兼容
- 保留 `circuit_breaker_losses/bars` 作为简单版后备
- 新增 `cb2_enabled` 开关
- 新增 `cb2_window_trades`, `cb2_l1/l2/l3_threshold`
