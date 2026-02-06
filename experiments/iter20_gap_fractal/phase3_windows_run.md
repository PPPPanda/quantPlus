# Phase 3: Windows 侧回测指南

vnpy 需要 Windows 环境运行。请在 PowerShell 中执行：

## 方法1: 直接运行测试脚本

```powershell
cd E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus
uv run python experiments/iter20_gap_fractal/phase3_test_s27.py
```

## 方法2: 手动运行单合约测试

```powershell
cd E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus

# 基线测试
uv run python -c "
from qp.backtest.engine import run_backtest
from vnpy.trader.constant import Interval
from datetime import datetime

result = run_backtest(
    'p2209.DCE',
    datetime(2020, 1, 1),
    datetime(2030, 1, 1),
    csv_path='data/analyse/wind/p2209_1min_202111-202209.csv',
    setting={
        'circuit_breaker_losses': 7,
        'circuit_breaker_bars': 70,
        'div_threshold': 0.39,
        'max_pullback_atr': 3.2,
        'gap_reset_inclusion': False,
    },
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)
print(f'PnL: {result.get(\"total_net_pnl\", 0):.1f}')
"

# S27 启用测试
uv run python -c "
from qp.backtest.engine import run_backtest
from vnpy.trader.constant import Interval
from datetime import datetime

result = run_backtest(
    'p2209.DCE',
    datetime(2020, 1, 1),
    datetime(2030, 1, 1),
    csv_path='data/analyse/wind/p2209_1min_202111-202209.csv',
    setting={
        'circuit_breaker_losses': 7,
        'circuit_breaker_bars': 70,
        'div_threshold': 0.39,
        'max_pullback_atr': 3.2,
        'gap_reset_inclusion': True,  # 启用 S27
    },
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)
print(f'PnL with S27: {result.get(\"total_net_pnl\", 0):.1f}')
"
```

## 预期结果

根据 Phase 2 分析，S27 启用应该：
- 改善 p2209 的跳空后表现（节假日一致率从 50% 预期提升）
- 对 p2401/p2601 影响较小（它们节假日一致率已经 100%）
- TOTAL 不应显著下降

## 后续步骤

回测完成后，比较：
1. BASELINE vs S27 的 TOTAL
2. 每个合约的 PnL 变化
3. 重点关注 p2209 的改善幅度
