"""verify_strategy_backtest.py

使用实际策略文件 (CtaChanPivotStrategy) 通过 vnpy BacktestingEngine 回测 7 合约,
对比 iteration_v6.py 脚本回测结果, 验证一致性.

运行方式 (Windows, 在项目 .venv 中):
  cd E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus
  .venv\Scripts\python.exe scripts\verify_strategy_backtest.py
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy


DATA_DIR = Path(__file__).parent.parent / "data" / "analyse"
CONTRACTS = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

# S6_macd15_3.0 参数 (这些是策略类的默认值, 确认一下)
S6_PARAMS = {
    "atr_trailing_mult": 2.0,
    "atr_activate_mult": 1.5,
    "atr_entry_filter": 1.5,
    "min_bi_gap": 5,
    "trend_filter": True,
    "disable_3s_short": True,
    "bi_amp_filter": True,
    "bi_amp_min_atr": 1.5,
    "macd_consistency": 3,
    "macd15_mag_cap_atr": 3.0,
}


def load_1m_data(contract: str) -> pd.DataFrame:
    """加载 1 分钟数据."""
    pattern = DATA_DIR / f"{contract}*.csv"
    files = sorted(DATA_DIR.glob(f"{contract}*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV for {contract} in {DATA_DIR}")
    fp = files[0]
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def run_vnpy_backtest(contract: str, df: pd.DataFrame) -> dict:
    """通过 vnpy BacktestingEngine 回测."""
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange

    engine = BacktestingEngine()

    # 转为 python datetime (避免 pandas Timestamp 导致 SQLite 绑定错误)
    start_dt = df["datetime"].iloc[0].to_pydatetime()
    end_dt = df["datetime"].iloc[-1].to_pydatetime()

    engine.set_parameters(
        vt_symbol=f"{contract}.SHFE",
        interval=Interval.MINUTE,
        start=start_dt,
        end=end_dt,
        rate=0.0,      # 不考虑手续费
        slippage=0,     # 不考虑滑点
        size=1,         # 每点价值
        pricetick=1,    # 最小变动价位
        capital=100000,
    )

    # 设置 S6 参数
    setting = dict(S6_PARAMS)
    setting["fixed_volume"] = 1

    engine.add_strategy(CtaChanPivotStrategy, setting)

    # 直接构造 history_data, 绕过数据库加载
    engine.history_data = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=contract,
            exchange=Exchange.SHFE,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=0,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="BACKTEST",
        )
        engine.history_data.append(bar)

    engine.run_backtesting()

    # 必须先 calculate_result() 生成逐日盈亏, 再 calculate_statistics()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    # 额外从 engine 直接取交易记录作为补充
    all_trades = engine.get_all_trades()
    trade_count = len(all_trades)
    if trade_count > 0 and (not stats or stats.get("total_count", 0) == 0):
        # fallback: 从 trade 列表手动计算
        net_pnl = sum(
            t.price * t.volume * (1 if t.direction.value == "LONG" else -1)
            for t in all_trades
        )
        print(f"  [DEBUG] engine.trades={trade_count}, stats={stats}")

    return {
        "contract": contract.upper(),
        "total_pnl": stats.get("total_net_pnl", 0) if stats else 0,
        "total_trades": stats.get("total_trade_count", 0) if stats else 0,
        "win_rate": 0,  # vnpy 不直接输出 win_rate, 后续从 trades 计算
        "max_drawdown": stats.get("max_drawdown", 0) if stats else 0,
        "engine_trade_count": trade_count,
        "stats": stats,
    }


def run_script_backtest(contract: str, df: pd.DataFrame) -> dict:
    """用 iteration_v6.py 的独立 tester 回测（作为对照）."""
    # 动态导入
    sys.path.insert(0, str(Path(__file__).parent))
    from iteration_v6 import ChanPivotTesterV7, StrategyParams

    p = StrategyParams(
        name="S6_verify",
        activate_atr=1.5,
        trail_atr=2.0,
        entry_filter_atr=1.5,
        min_bi_gap=5,
        trend_filter=True,
        disable_3s_short=True,
        bi_amp_filter=True,
        bi_amp_min_atr=1.5,
        macd_consistency=3,
        macd15_mag_cap_atr=3.0,
    )

    tester = ChanPivotTesterV7(df, p)
    trades_df = tester.run()

    if trades_df.empty:
        return {"contract": contract.upper(), "total_pnl": 0, "total_trades": 0, "win_rate": 0}

    total_pnl = trades_df["pnl"].sum()
    wins = (trades_df["pnl"] > 0).sum()
    n = len(trades_df)

    return {
        "contract": contract.upper(),
        "total_pnl": total_pnl,
        "total_trades": n,
        "win_rate": wins / n * 100 if n > 0 else 0,
    }


def main():
    print("=" * 80)
    print("VERIFY: Strategy File vs Script Backtest Consistency")
    print("Config: S6_macd15_3.0")
    print("=" * 80)

    results_vnpy = []
    results_script = []

    for contract in CONTRACTS:
        print(f"\n--- {contract.upper()} ---")
        df = load_1m_data(contract)
        print(f"  Loaded {len(df)} 1min bars ({df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")

        # Script backtest (always works)
        print("  Running script backtest...")
        r_script = run_script_backtest(contract, df)
        results_script.append(r_script)
        print(f"  Script: PnL={r_script['total_pnl']:.1f}, Trades={r_script['total_trades']}, Win={r_script['win_rate']:.1f}%")

        # vnpy backtest
        try:
            print("  Running vnpy backtest...")
            r_vnpy = run_vnpy_backtest(contract, df)
            results_vnpy.append(r_vnpy)
            print(f"  vnpy:   PnL={r_vnpy['total_pnl']:.1f}, Trades={r_vnpy['total_trades']}, Win={r_vnpy['win_rate']:.1f}%")

            # Compare
            pnl_diff = abs(r_vnpy['total_pnl'] - r_script['total_pnl'])
            if pnl_diff < 1.0:
                print(f"  MATCH OK (diff={pnl_diff:.2f})")
            else:
                print(f"  *** MISMATCH *** diff={pnl_diff:.2f}")
        except Exception as e:
            print(f"  vnpy backtest failed: {e}")
            results_vnpy.append({"contract": contract.upper(), "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Contract':>10} | {'Script PnL':>12} | {'vnpy PnL':>12} | {'Diff':>10} | {'Status':>8}")
    print("-" * 70)
    total_script = 0
    total_vnpy = 0
    all_match = True
    for i, rs in enumerate(results_script):
        c = rs['contract']
        sp = rs['total_pnl']
        total_script += sp
        if i < len(results_vnpy) and 'error' not in results_vnpy[i]:
            vp = results_vnpy[i]['total_pnl']
            total_vnpy += vp
            diff = abs(sp - vp)
            status = "OK" if diff < 1.0 else "MISMATCH"
            if diff >= 1.0:
                all_match = False
            print(f"{c:>10} | {sp:>12.1f} | {vp:>12.1f} | {diff:>10.2f} | {status:>8}")
        else:
            err = results_vnpy[i].get('error', '?') if i < len(results_vnpy) else 'N/A'
            print(f"{c:>10} | {sp:>12.1f} | {'ERROR':>12} | {'N/A':>10} | {'ERROR':>8}")
            all_match = False

    print("-" * 70)
    print(f"{'TOTAL':>10} | {total_script:>12.1f} | {total_vnpy:>12.1f} | {abs(total_script-total_vnpy):>10.2f} |")
    print()
    if all_match:
        print("ALL CONTRACTS MATCH - Strategy file is consistent with script backtest")
    else:
        print("WARNING: Some contracts have mismatches - investigation needed")

    # Save results
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": S6_PARAMS,
        "script_results": results_script,
        "vnpy_results": results_vnpy,
        "all_match": all_match,
    }
    outpath = Path(__file__).parent.parent / "experiments" / "s6_verification.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
