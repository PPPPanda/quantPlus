"""
Phase 2：失败模式分解 — 逐笔交易诊断脚本.

读取 ChanDebugger 输出的 trades.csv + signals.csv，
分析每笔交易的信号类型、盈亏、中枢关系等。

用法：
    cd quantPlus
    .venv/Scripts/python.exe scripts/diagnose_trades.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEBUG_DIR = ROOT / "data" / "debug"


def find_latest_debug_dirs(contracts: list[str]) -> dict[str, Path]:
    """找每个合约的最新 debug 目录."""
    results = {}
    for d in sorted(DEBUG_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        name = d.name  # e.g. CtaChanPivotStrategy_20260204_012540
        for c in contracts:
            if c not in results:
                # 检查 summary.json 里的合约
                summary_path = d / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                        if c in str(summary.get("config", {}).get("vt_symbol", "")):
                            results[c] = d
                    except Exception:
                        pass
                # 也通过 trades 文件判断
                trades_path = d / "trades.csv"
                if trades_path.exists() and c not in results:
                    results[c] = d
    return results


def analyze_contract(debug_dir: Path, contract: str) -> dict:
    """分析单合约的交易诊断."""
    trades_path = debug_dir / "trades.csv"
    signals_path = debug_dir / "signals.csv"
    summary_path = debug_dir / "summary.json"

    result = {"contract": contract, "debug_dir": str(debug_dir.name)}

    # 读取交易记录
    if not trades_path.exists():
        result["error"] = "trades.csv not found"
        return result

    trades_df = pd.read_csv(trades_path)
    if trades_df.empty:
        result["error"] = "no trades"
        return result

    # 信号记录
    signals_df = None
    if signals_path.exists():
        signals_df = pd.read_csv(signals_path)

    # 读取 summary
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    # --- 基础统计 ---
    # 按信号类型分组
    if "signal_type" in trades_df.columns:
        # 按 open/close 配对
        opens = trades_df[trades_df["action"].str.contains("OPEN")]
        closes = trades_df[trades_df["action"].str.contains("CLOSE")]

        # 配对交易（简单：按顺序配对）
        paired = []
        for i in range(min(len(opens), len(closes))):
            o = opens.iloc[i]
            c = closes.iloc[i]
            pnl = c["pnl"] if "pnl" in c.index else 0
            paired.append({
                "open_time": o.get("timestamp", ""),
                "close_time": c.get("timestamp", ""),
                "signal_type": o.get("signal_type", "unknown"),
                "direction": "long" if "LONG" in str(o["action"]) else "short",
                "entry_price": o["price"],
                "exit_price": c["price"],
                "pnl": pnl,
            })

        paired_df = pd.DataFrame(paired)
        if not paired_df.empty:
            # 按信号类型分组统计
            by_signal = paired_df.groupby("signal_type").agg(
                count=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
                max_loss=("pnl", "min"),
                max_win=("pnl", "max"),
            ).round(2)

            result["by_signal_type"] = by_signal.to_dict("index")

            # 按方向分组
            by_dir = paired_df.groupby("direction").agg(
                count=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
            ).round(2)

            result["by_direction"] = by_dir.to_dict("index")

            # 找 Top 3 最大亏损交易
            worst = paired_df.nsmallest(3, "pnl")
            result["worst_trades"] = worst.to_dict("records")

            # 总统计
            result["total_trades"] = len(paired_df)
            result["total_pnl_points"] = round(paired_df["pnl"].sum(), 2)
            result["win_rate"] = round((paired_df["pnl"] > 0).mean(), 4)
            result["avg_pnl"] = round(paired_df["pnl"].mean(), 2)

    # 信号统计
    if signals_df is not None and not signals_df.empty:
        result["total_signals"] = len(signals_df)
        if "signal_type" in signals_df.columns:
            result["signals_by_type"] = signals_df["signal_type"].value_counts().to_dict()

    # summary 摘要
    if summary:
        chan = summary.get("chan_stats", {})
        result["chan_stats"] = {
            "total_bi": chan.get("total_bi", 0),
            "total_pivots": chan.get("total_pivots", 0),
            "bi_per_pivot": round(chan.get("total_bi", 0) / max(chan.get("total_pivots", 1), 1), 2),
        }

    return result


def main():
    contracts = ["p2601", "p2405", "p2209"]

    print("Phase 2: 失败模式分解 — 交易诊断")
    print("=" * 60)

    # 找 debug 目录
    debug_dirs = find_latest_debug_dirs(contracts)

    all_results = []
    for c in contracts:
        if c in debug_dirs:
            print(f"\n--- {c} ({debug_dirs[c].name}) ---")
            r = analyze_contract(debug_dirs[c], c)
            all_results.append(r)
            
            # 打印关键信息
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue

            print(f"  Trades: {r.get('total_trades', '?')}")
            print(f"  Total PnL (points): {r.get('total_pnl_points', '?')}")
            print(f"  Win Rate: {r.get('win_rate', '?')}")

            if "by_signal_type" in r:
                print(f"  By Signal Type:")
                for sig, stats in r["by_signal_type"].items():
                    print(f"    {sig}: count={stats['count']}, pnl={stats['total_pnl']:.0f}, "
                          f"win={stats['win_rate']:.0%}, avg={stats['avg_pnl']:.0f}")

            if "by_direction" in r:
                print(f"  By Direction:")
                for d, stats in r["by_direction"].items():
                    print(f"    {d}: count={stats['count']}, pnl={stats['total_pnl']:.0f}, "
                          f"win={stats['win_rate']:.0%}")

            if "worst_trades" in r:
                print(f"  Worst 3 Trades:")
                for t in r["worst_trades"]:
                    print(f"    {t['signal_type']} {t['direction']}: pnl={t['pnl']:.0f} "
                          f"({t.get('open_time', '?')} -> {t.get('close_time', '?')})")
        else:
            print(f"\n--- {c}: no debug dir found ---")
            all_results.append({"contract": c, "error": "no debug dir"})

    # 保存
    out_path = ROOT / "experiments" / "iter1" / "phase2_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
