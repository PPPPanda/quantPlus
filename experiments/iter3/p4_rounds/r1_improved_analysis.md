# R1-Improved Grid Search Analysis

## Date: 2025-02-04

## Summary
Grid search tested 31 combinations of divergence modes on 7 contracts.

**Best config: Mode 1 (OR logic), threshold=0.70, min_areas=2**

## Results Overview

| Mode | Description | Combos | PASS | Best pts | Best n_neg |
|------|-------------|--------|------|----------|------------|
| 0 | Baseline (original diff) | 1 | 1 | 8244.0 | 3 |
| 1 | diff_ok OR divergence | 10 | 10 | 9475.2 | 1 |
| 2 | diff_ok AND divergence | 10 | 0 | 304.2 | 3 |
| 3 | divergence-only | 10 | 0 | 659.0 | 3 |

## Best Config vs Baseline (per contract)

| Contract | BL pnl | M1 pnl | Delta | BL Sharpe | M1 Sharpe |
|----------|--------|--------|-------|-----------|-----------|
| p2601 | +14,932 | +12,258 | -2,674 | 3.43 | 2.65 |
| p2405 | +5,729 | +7,558 | +1,829 | 1.71 | 2.21 |
| p2209 | +67,362 | +70,710 | +3,347 | 4.14 | 4.33 |
| p2501 | -764 | +2,198 | +2,962 | -0.11 | 0.32 |
| p2505 | +1,934 | +4,941 | +3,007 | 0.65 | 1.52 |
| p2509 | -435 | +915 | +1,350 | -0.23 | 0.41 |
| p2401 | -6,318 | -3,828 | +2,490 | -4.18 | -2.27 |
| **TOTAL** | **82,440** | **94,752** | **+12,313** | | |
| **neg** | **3** | **1** | **-2** | | |

## Key Findings

1. **Mode 1 (OR) is a clear winner**: +12,313 pnl improvement, negative contracts reduced from 3→1
2. **p2501 flipped positive** (+2,198 from -764): biggest relative improvement
3. **p2509 flipped positive** (+915 from -435): second flip
4. **p2401 improved but still negative** (-3,828 from -6,318): improved by 2,490 but structurally problematic
5. **p2601 slightly regressed** (-2,674): small trade-off for overall improvement
6. **min_areas parameter has zero impact**: a=2 and a=3 produce identical results for all thresholds
7. **Mode 2 (AND) is catastrophic**: all FAIL, worst at -159 pts
8. **Mode 3 (divergence-only) also fails**: all FAIL, max 659 pts

## Threshold Sensitivity (Mode 1)

| Threshold | pts | n_neg | p2401 pnl |
|-----------|-----|-------|-----------|
| 0.70 | 9475.2 | 1 | -3827.8 |
| 0.80 | 9403.4 | 1 | -3827.8 |
| 0.90 | 9379.5 | 1 | -3836.5 |
| 0.95 | 9355.9 | 1 | -3836.5 |
| 1.00 | 9355.9 | 1 | -3836.5 |

Lower threshold (0.70) is best — makes divergence condition easier to satisfy, generating more signals.

## Recommendation
- **Adopt Mode 1 (t=0.70)** as new baseline
- **Drop min_areas parameter** (no effect)
- **p2401 needs separate investigation** — likely structural issue (market regime, not strategy logic)
- Next: formalize monkey-patch into strategy code, then tackle backlog items (R2 stop-loss, R3 pivot state machine)
