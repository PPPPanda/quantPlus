"""Phase 2: 逐笔交易诊断 — 输出每笔交易的结构状态快照."""
from __future__ import annotations
import sys, json, logging
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from run_7bench import BENCHMARKS, BT_PARAMS, import_csv_to_db
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "div_mode": 1,
    "stop_buffer_atr_pct": 0.02,
}


def diagnose_contract(bench: dict) -> dict:
    """Run backtest and extract trade-level diagnostics."""
    vt = bench["contract"]
    name = vt.split(".")[0]
    start, end, _ = import_csv_to_db(bench["csv"], vt)
    r = run_backtest(
        vt_symbol=vt,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=dict(DEFAULT_SETTING),
        **BT_PARAMS,
    )

    s = r.stats or {}
    trades = r.trades or []

    # Group trades into round-trips (open + close pairs)
    round_trips = []
    i = 0
    while i < len(trades) - 1:
        t_open = trades[i]
        t_close = trades[i + 1]

        # Verify it's an open-close pair
        is_open = "OPEN" in str(t_open.offset).upper()
        is_close = "CLOSE" in str(t_close.offset).upper()

        if is_open and is_close:
            pnl_pts = 0
            if "LONG" in str(t_open.direction).upper():
                pnl_pts = (t_close.price - t_open.price) / 1.0  # price is already in points
            else:
                pnl_pts = (t_open.price - t_close.price) / 1.0

            duration_bars = 0
            if t_open.datetime and t_close.datetime:
                delta = t_close.datetime - t_open.datetime
                duration_bars = int(delta.total_seconds() / 60)  # 1m bars

            round_trips.append({
                "open_time": str(t_open.datetime),
                "close_time": str(t_close.datetime),
                "direction": str(t_open.direction),
                "open_price": float(t_open.price),
                "close_price": float(t_close.price),
                "pnl_pts": round(pnl_pts, 1),
                "duration_min": duration_bars,
                "win": pnl_pts > 0,
            })
            i += 2
        else:
            i += 1

    # Classify round trips
    wins = [rt for rt in round_trips if rt["win"]]
    losses = [rt for rt in round_trips if not rt["win"]]

    # Sort losses by magnitude
    losses_sorted = sorted(losses, key=lambda x: x["pnl_pts"])

    # Time-based clustering: group by week
    weekly = {}
    for rt in round_trips:
        week = rt["open_time"][:10]  # date
        if week not in weekly:
            weekly[week] = {"count": 0, "pnl": 0}
        weekly[week]["count"] += 1
        weekly[week]["pnl"] = round(weekly[week]["pnl"] + rt["pnl_pts"], 1)

    # Find worst streaks (consecutive losses)
    streaks = []
    current_streak = []
    for rt in round_trips:
        if not rt["win"]:
            current_streak.append(rt)
        else:
            if len(current_streak) >= 3:
                streak_pnl = sum(r["pnl_pts"] for r in current_streak)
                streaks.append({
                    "length": len(current_streak),
                    "pnl_pts": round(streak_pnl, 1),
                    "start": current_streak[0]["open_time"],
                    "end": current_streak[-1]["close_time"],
                })
            current_streak = []
    if len(current_streak) >= 3:
        streak_pnl = sum(r["pnl_pts"] for r in current_streak)
        streaks.append({
            "length": len(current_streak),
            "pnl_pts": round(streak_pnl, 1),
            "start": current_streak[0]["open_time"],
            "end": current_streak[-1]["close_time"],
        })

    # Duration analysis
    durations = [rt["duration_min"] for rt in round_trips]
    avg_dur = sum(durations) / len(durations) if durations else 0
    win_durations = [rt["duration_min"] for rt in wins]
    loss_durations = [rt["duration_min"] for rt in losses]
    avg_win_dur = sum(win_durations) / len(win_durations) if win_durations else 0
    avg_loss_dur = sum(loss_durations) / len(loss_durations) if loss_durations else 0

    # PnL distribution
    pnls = [rt["pnl_pts"] for rt in round_trips]
    avg_win = sum(r["pnl_pts"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0

    # Find worst 5 daily PnL
    worst_days = sorted(weekly.items(), key=lambda x: x[1]["pnl"])[:5]

    result = {
        "contract": name,
        "summary": {
            "total_pnl_pts": round(sum(pnls), 1),
            "total_trades": len(round_trips),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(round_trips) * 100, 1) if round_trips else 0,
            "avg_win_pts": round(avg_win, 1),
            "avg_loss_pts": round(avg_loss, 1),
            "profit_factor": round(abs(avg_win * len(wins) / (avg_loss * len(losses))), 2) if losses and avg_loss != 0 else 999,
            "avg_duration_min": round(avg_dur, 0),
            "avg_win_duration_min": round(avg_win_dur, 0),
            "avg_loss_duration_min": round(avg_loss_dur, 0),
        },
        "worst_losses": losses_sorted[:10],
        "worst_streaks": sorted(streaks, key=lambda x: x["pnl_pts"])[:5],
        "worst_days": [{"date": d, **v} for d, v in worst_days],
        "short_trades": len([rt for rt in round_trips if rt["duration_min"] <= 5]),
        "very_short_trades": len([rt for rt in round_trips if rt["duration_min"] <= 2]),
    }

    return result


if __name__ == "__main__":
    all_diag = {}
    for bench in BENCHMARKS:
        name = bench["contract"].split(".")[0]
        print(f"\n{'='*60}")
        print(f"Diagnosing: {name}")
        print(f"{'='*60}")

        diag = diagnose_contract(bench)
        all_diag[name] = diag

        s = diag["summary"]
        print(f"  PnL: {s['total_pnl_pts']:+.1f} pts | Trades: {s['total_trades']} | "
              f"Win%: {s['win_rate']}% | PF: {s['profit_factor']}")
        print(f"  Avg win: +{s['avg_win_pts']:.1f} | Avg loss: {s['avg_loss_pts']:.1f} | "
              f"Duration: {s['avg_duration_min']:.0f}min (win:{s['avg_win_duration_min']:.0f} loss:{s['avg_loss_duration_min']:.0f})")
        print(f"  Short trades (≤5min): {diag['short_trades']} | Very short (≤2min): {diag['very_short_trades']}")

        if diag["worst_streaks"]:
            print(f"  Worst streak: {diag['worst_streaks'][0]['length']} losses, "
                  f"{diag['worst_streaks'][0]['pnl_pts']:+.1f} pts")
        if diag["worst_days"]:
            print(f"  Worst day: {diag['worst_days'][0]['date']} = "
                  f"{diag['worst_days'][0]['pnl']:+.1f} pts")

    # Save
    out = ROOT / "experiments/iter4/phase2_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_diag, f, indent=2, default=str)
    print(f"\nSaved: {out}")
