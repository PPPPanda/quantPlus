"""
Phase 2：3 合约交易诊断分析.

读取最新 debug 目录的 trades.csv / signals.csv，输出诊断报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = ROOT / "data" / "debug"

# 按 bars 数映射到合约
BARS_TO_CONTRACT = {
    30231: "p2601.DCE",
    28502: "p2405.DCE",
    28408: "p2209.DCE",
}

def find_debug_dirs() -> dict[str, Path]:
    """找最新的3合约 debug 目录（按时间戳倒序）."""
    results = {}
    for d in sorted(DEBUG_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("CtaChanPivotStrategy_20260204"):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        bars = summary.get("total_bars_1m", 0)
        contract = BARS_TO_CONTRACT.get(bars)
        if contract and contract not in results:
            results[contract] = d
        if len(results) == 3:
            break
    return results


def analyze_trades(debug_dir: Path, contract: str) -> dict:
    """诊断单合约交易."""
    trades = pd.read_csv(debug_dir / "trades.csv")
    signals = pd.read_csv(debug_dir / "signals.csv") if (debug_dir / "signals.csv").exists() else None
    summary = json.loads((debug_dir / "summary.json").read_text())

    # 配对开平仓
    opens = trades[trades["action"].str.contains("OPEN")].reset_index(drop=True)
    closes = trades[trades["action"].str.contains("CLOSE")].reset_index(drop=True)

    paired = []
    for i in range(min(len(opens), len(closes))):
        o = opens.iloc[i]
        c = closes.iloc[i]
        direction = "long" if "LONG" in o["action"] else "short"
        paired.append({
            "signal_type": o["signal_type"],
            "direction": direction,
            "entry": o["price"],
            "exit": c["price"],
            "pnl": c["pnl"],
        })

    pdf = pd.DataFrame(paired)

    # === 按信号类型 ===
    by_sig = {}
    for sig in ["3B", "3S", "2B", "2S"]:
        sub = pdf[pdf["signal_type"] == sig]
        if len(sub) == 0:
            continue
        by_sig[sig] = {
            "count": len(sub),
            "total_pnl": round(sub["pnl"].sum(), 1),
            "avg_pnl": round(sub["pnl"].mean(), 1),
            "win_rate": round((sub["pnl"] > 0).mean(), 3),
            "max_win": round(sub["pnl"].max(), 1),
            "max_loss": round(sub["pnl"].min(), 1),
        }

    # === 按方向 ===
    by_dir = {}
    for d in ["long", "short"]:
        sub = pdf[pdf["direction"] == d]
        if len(sub) == 0:
            continue
        by_dir[d] = {
            "count": len(sub),
            "total_pnl": round(sub["pnl"].sum(), 1),
            "avg_pnl": round(sub["pnl"].mean(), 1),
            "win_rate": round((sub["pnl"] > 0).mean(), 3),
        }

    # === 交叉：信号×方向 ===
    cross = {}
    for sig in ["3B", "3S", "2B", "2S"]:
        for d in ["long", "short"]:
            sub = pdf[(pdf["signal_type"] == sig) & (pdf["direction"] == d)]
            if len(sub) == 0:
                continue
            cross[f"{sig}_{d}"] = {
                "count": len(sub),
                "total_pnl": round(sub["pnl"].sum(), 1),
                "win_rate": round((sub["pnl"] > 0).mean(), 3),
            }

    # === Top 5 最大亏损 ===
    worst = pdf.nsmallest(5, "pnl")[["signal_type", "direction", "entry", "exit", "pnl"]].to_dict("records")

    # === 连亏分析 ===
    streaks = []
    current_streak = 0
    current_loss = 0
    for _, row in pdf.iterrows():
        if row["pnl"] < 0:
            current_streak += 1
            current_loss += row["pnl"]
        else:
            if current_streak >= 3:
                streaks.append({"length": current_streak, "total_loss": round(current_loss, 1)})
            current_streak = 0
            current_loss = 0
    if current_streak >= 3:
        streaks.append({"length": current_streak, "total_loss": round(current_loss, 1)})

    return {
        "contract": contract,
        "debug_dir": debug_dir.name,
        "total_trades": len(pdf),
        "internal_pnl": round(pdf["pnl"].sum(), 1),
        "win_rate": round((pdf["pnl"] > 0).mean(), 3),
        "avg_pnl": round(pdf["pnl"].mean(), 1),
        "by_signal_type": by_sig,
        "by_direction": by_dir,
        "cross_signal_direction": cross,
        "worst_5_trades": worst,
        "losing_streaks_3plus": streaks,
        "chan_stats": {
            "total_bi": summary.get("total_bi", 0),
            "total_pivot": summary.get("total_pivot", 0),
            "total_signals": summary.get("total_signals", 0),
            "total_trades_debug": summary.get("total_trades", 0),
        },
    }


def main():
    print("Phase 2: Trade Diagnosis")
    print("=" * 70)

    dirs = find_debug_dirs()
    all_results = []

    for contract in ["p2601.DCE", "p2405.DCE", "p2209.DCE"]:
        if contract not in dirs:
            print(f"\n{contract}: NOT FOUND")
            continue

        r = analyze_trades(dirs[contract], contract)
        all_results.append(r)

        slot = {"p2601.DCE": "BENCH", "p2405.DCE": "FIX", "p2209.DCE": "STRESS"}[contract]
        print(f"\n{'='*70}")
        print(f"[{slot}] {contract}  (internal PnL = {r['internal_pnl']} pts)")
        print(f"  Trades: {r['total_trades']}  WinRate: {r['win_rate']:.1%}  AvgPnL: {r['avg_pnl']}")
        print(f"  Chan: {r['chan_stats']['total_bi']} bi, {r['chan_stats']['total_pivot']} pivots, "
              f"{r['chan_stats']['total_signals']} signals")

        print(f"\n  By Signal Type:")
        for sig, s in r["by_signal_type"].items():
            print(f"    {sig:3s}: {s['count']:3d} trades, pnl={s['total_pnl']:+8.1f}, "
                  f"win={s['win_rate']:.0%}, avg={s['avg_pnl']:+.1f}, "
                  f"best={s['max_win']:+.1f}, worst={s['max_loss']:+.1f}")

        print(f"\n  By Direction:")
        for d, s in r["by_direction"].items():
            print(f"    {d:5s}: {s['count']:3d} trades, pnl={s['total_pnl']:+8.1f}, "
                  f"win={s['win_rate']:.0%}")

        print(f"\n  Cross (signal x direction):")
        for k, s in r["cross_signal_direction"].items():
            print(f"    {k:10s}: {s['count']:3d} trades, pnl={s['total_pnl']:+8.1f}, win={s['win_rate']:.0%}")

        print(f"\n  Worst 5 Trades:")
        for t in r["worst_5_trades"]:
            print(f"    {t['signal_type']} {t['direction']:5s}: pnl={t['pnl']:+.1f} "
                  f"(entry={t['entry']:.0f} -> exit={t['exit']:.0f})")

        if r["losing_streaks_3plus"]:
            print(f"\n  Losing Streaks (>=3):")
            for s in r["losing_streaks_3plus"]:
                print(f"    {s['length']} consecutive losses, total={s['total_loss']:.1f}")

    # 保存
    out = ROOT / "experiments" / "iter1" / "phase2_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nSaved: {out}")


if __name__ == "__main__":
    main()
