# iter10 总结

## 基线 (iter9)
- TOTAL=11551.1 pts
- 参数: cb=7/70, cd=2/20, dt=0.50

## 研究方向
1. **止盈/锁盈**: lock_profit_atr 全系列失败，与策略逻辑不兼容
2. **参数微调**: min_hold, pullback, entry_filter 均无正面效果
3. **div_threshold 优化**: ✅ 发现 dt=0.40 显著改善

## 实验记录

### 止盈方向
| 配置 | TOTAL | p2301 | Neg |
|------|-------|-------|-----|
| lock=0.3 | 12273 | -162 | p2301 |
| lock=0.5 | 5158 | -72 | p2301 |
| lock=1.0 | 12243 | -22 | p2301 |

### 参数微调
| 配置 | TOTAL | p2301 | Neg |
|------|-------|-------|-----|
| min_hold=3 | 4149 | 65 | None |
| pullback=3.5 | 11774 | -129 | p2301 |
| pullback=5.0 | 11380 | 86 | None |
| entry_filter=1.5 | 10637 | -148 | p2301 |
| entry_filter=2.5 | 11508 | 109 | None |
| stop_buffer=0.03 | 11551 | 109 | None |
| pivot_valid_range=8 | 11551 | 109 | None |

### div_threshold 扫描 ✅
| dt | TOTAL | p2209 | p2301 | p2509 | Neg |
|----|-------|-------|-------|-------|-----|
| 0.30 | 11147 | 7058 | 222 | 147 | None |
| 0.35 | 11354 | 7058 | 379 | 147 | None |
| **0.40** | **11965** | **7490** | **397** | **147** | **None** |
| 0.45 | 11874 | 7440 | 397 | 155 | None |
| 0.50 | 11551 | 7440 | 109 | 147 | None |
| 0.60 | 11125 | 7350 | 42 | -11 | p2509 |

## 采纳变更
- `div_threshold`: 0.50 → **0.40**

## 效果（vs iter9 基线）
- TOTAL: 11551.1 → **11964.6** (+413.5 pts, +3.6%)
- p2301: 109.3 → **396.6** (+287.3, 最大受益者)
- p2209: 7439.6 → 7489.5 (+49.9)
- p2501: 1148.1 → 1203.2 (+55.1)
- p2601: 1260.7 → 1264.2 (+3.5)
- p2505: 558.0 → 575.8 (+17.8)
- p2509: 146.9 → 146.9 (0)

## 效果（vs iter8 原始基线）
- TOTAL: 11117.3 → **11964.6** (+847.3 pts, +7.6%)

## 新基线 (iter10)
```
circuit_breaker_losses=7, circuit_breaker_bars=70
cooldown_losses=2, cooldown_bars=20
min_hold_bars=2, max_pullback_atr=4.0
div_mode=1, div_threshold=0.40
atr_activate_mult=2.5, atr_trailing_mult=3.0, atr_entry_filter=2.0
use_bi_trailing=True
seg_enabled=False, hist_gate=0
```
TOTAL = 11964.6 pts

## 关键发现
1. **止盈机制不适合趋势跟踪策略**: 当前策略的大赢家来自长期持仓，过早止盈/锁盈会显著降低盈利
2. **div_threshold 是敏感参数**: 0.40 是当前最优，过低（<0.35）限制入场过多，过高（>0.55）允许太多噪声信号
3. **弱合约 p2301 对 2B 信号质量极度敏感**: dt从0.50→0.40提升p2301近4倍
