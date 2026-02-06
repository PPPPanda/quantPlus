# iter9 总结

## 基线
- TOTAL=11117.3 pts, 全合约正收益
- 参数: cb=6/60, cd=2/20

## 研究方向
1. **节假日保护**: 审计完成。Overnight 持仓贡献 62% 利润（6868/11117 pts），节假日保护弊大于利。不实施。
2. **止盈策略**: 审计完成。当前 trailing stop 已是自适应止盈，加入固定目标会砍大赢家。不实施。
3. **参数优化**: 采纳 cb=7/70。

## 实验记录

| 配置 | TOTAL | p2209 | p2301 | p2509 | Neg |
|------|-------|-------|-------|-------|-----|
| **baseline cb=6/60** | **11117.3** | 6937.6 | 175.0 | 146.9 | None |
| cb=7/60 | 11537.3 | 7439.6 | 109.3 | 146.9 | None |
| **cb=7/70** | **11551.1** | **7439.6** | **109.3** | **146.9** | **None** |
| cb=7/80 | 11551.1 | 7439.6 | 109.3 | 146.9 | None |
| cb=7/70+cd2/15 | 11370.9 | 7689.8 | -41.5 | -68.8 | p2509,p2301 |
| cb=7/70+cd2/25 | 11396.0 | 7389.6 | 120.2 | 66.2 | None |

## 采纳变更
- `circuit_breaker_losses`: 6 → **7**
- `circuit_breaker_bars`: 60 → **70**

## 效果
- TOTAL: 11117.3 → **11551.1** (+433.8 pts, +3.9%)
- p2209: 6937.6 → 7439.6 (+502)
- p2601: 1246.9 → 1260.7 (+13.8)
- p2301: 175.0 → 109.3 (-65.7) ⚠️ 仍正
- 其他合约不变

## 新基线 (iter9)
```
circuit_breaker_losses=7, circuit_breaker_bars=70
cooldown_losses=2, cooldown_bars=20
min_hold_bars=2, max_pullback_atr=4.0
div_mode=1, div_threshold=0.50
atr_activate_mult=2.5, atr_trailing_mult=3.0, atr_entry_filter=2.0
use_bi_trailing=True
seg_enabled=False, hist_gate=0
```
TOTAL = 11551.1 pts

## 关键诊断发现（供 iter10 使用）
- Overnight PnL = 6868 pts (62% of total)
- 最大单笔亏损 p2209: -1760 pts, p2501: -1360 pts（跨周末/假日gap）
- 弱合约 WR 仅 35%，依赖少数大赢家
- 连亏簇长度 5-9 很常见
