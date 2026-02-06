"""诊断脚本：分析短持仓交易的失败特征."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

ROOT = Path(__file__).parent.parent

BENCHMARKS = [
    ("p2201.DCE", ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"),
    ("p2209.DCE", ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"),
    ("p2401.DCE", ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"),
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)


def import_csv_to_db(csv_path: Path, vt_symbol: str):
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    from zoneinfo import ZoneInfo

    CN_TZ = ZoneInfo("Asia/Shanghai")
    db = get_database()
    symbol, exchange_str = vt_symbol.split(".")
    exchange = Exchange(exchange_str)
    db.delete_bar_data(symbol, exchange, Interval.MINUTE)

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    df.sort_values("datetime", inplace=True)
    df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(CN_TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(CN_TZ)

    bars = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=symbol, exchange=exchange, datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="DB",
        )
        bars.append(bar)

    db.save_bar_data(bars)
    return df["datetime"].min(), df["datetime"].max(), df


def run_with_trade_log(vt_symbol, csv_path, setting):
    """跑回测并返回交易记录和价格数据."""
    start, end, df = import_csv_to_db(csv_path, vt_symbol)
    if hasattr(start, 'to_pydatetime'):
        start = start.to_pydatetime()
    if hasattr(end, 'to_pydatetime'):
        end = end.to_pydatetime()
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS
    )
    return result, df


def analyze_short_holds(result, df, contract):
    """分析短持仓交易特征."""
    trades = result.trades
    if not trades:
        return []
    
    # 配对交易
    short_holds = []
    open_trade = None
    
    for t in trades:
        if t.offset.value == "开":
            open_trade = t
        elif t.offset.value == "平" and open_trade:
            hold_minutes = (t.datetime - open_trade.datetime).total_seconds() / 60
            hold_bars = hold_minutes / 5
            
            if hold_bars <= 50:  # 只分析短持仓
                if open_trade.direction.value == "多":
                    pnl = (t.price - open_trade.price)
                else:
                    pnl = (open_trade.price - t.price)
                
                # 计算入场时的市场状态
                open_time = open_trade.datetime
                if hasattr(open_time, 'tz_localize'):
                    pass  # 已有时区
                
                # 找入场前的价格数据
                df_before = df[df['datetime'] <= open_time].tail(20)
                
                # 计算入场位置相对近期高低点的位置
                if len(df_before) >= 10:
                    recent_high = df_before['high'].max()
                    recent_low = df_before['low'].min()
                    price_range = recent_high - recent_low
                    if price_range > 0:
                        entry_pct = (open_trade.price - recent_low) / price_range
                    else:
                        entry_pct = 0.5
                else:
                    entry_pct = 0.5
                
                short_holds.append({
                    'contract': contract,
                    'open_time': str(open_trade.datetime),
                    'close_time': str(t.datetime),
                    'hold_bars': hold_bars,
                    'pnl': pnl,
                    'entry_price': open_trade.price,
                    'exit_price': t.price,
                    'entry_pct': entry_pct,  # 0=底部, 1=顶部
                    'is_winner': pnl > 0,
                })
            
            open_trade = None
    
    return short_holds


def main():
    setting = {
        "debug_enabled": False,
        "debug_log_console": False,
        "circuit_breaker_losses": 7,
        "circuit_breaker_bars": 70,
        "div_threshold": 0.39,
        "max_pullback_atr": 3.2,
    }
    
    all_short = []
    
    for contract, csv_path in BENCHMARKS:
        print(f"Processing {contract}...", flush=True)
        result, df = run_with_trade_log(contract, csv_path, setting)
        shorts = analyze_short_holds(result, df, contract)
        all_short.extend(shorts)
        print(f"  {len(shorts)} short holds")
    
    if not all_short:
        print("No short holds found!")
        return
    
    df = pd.DataFrame(all_short)
    
    print("\n" + "="*60)
    print("SHORT HOLD ANALYSIS (<= 50 bars)")
    print("="*60)
    
    # 按持仓时长分组
    bins = [0, 5, 10, 20, 50]
    labels = ['<5', '5-10', '10-20', '20-50']
    df['hold_group'] = pd.cut(df['hold_bars'], bins=bins, labels=labels)
    
    # 按入场位置分组
    df['entry_zone'] = pd.cut(df['entry_pct'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Mid', 'High'])
    
    print("\n1. By Holding Time:")
    for grp in labels:
        grp_df = df[df['hold_group'] == grp]
        if len(grp_df) > 0:
            win_rate = grp_df['is_winner'].mean() * 100
            total_pnl = grp_df['pnl'].sum()
            avg_pnl = grp_df['pnl'].mean()
            print(f"  {grp} bars: {len(grp_df)} trades, PnL={total_pnl:.0f}, WR={win_rate:.1f}%, Avg={avg_pnl:.1f}")
    
    print("\n2. By Entry Position (relative to recent 20-bar range):")
    for zone in ['Low', 'Mid', 'High']:
        zone_df = df[df['entry_zone'] == zone]
        if len(zone_df) > 0:
            win_rate = zone_df['is_winner'].mean() * 100
            total_pnl = zone_df['pnl'].sum()
            avg_pnl = zone_df['pnl'].mean()
            print(f"  {zone}: {len(zone_df)} trades, PnL={total_pnl:.0f}, WR={win_rate:.1f}%, Avg={avg_pnl:.1f}")
    
    print("\n3. By Contract:")
    for c in df['contract'].unique():
        c_df = df[df['contract'] == c]
        win_rate = c_df['is_winner'].mean() * 100
        total_pnl = c_df['pnl'].sum()
        print(f"  {c}: {len(c_df)} trades, PnL={total_pnl:.0f}, WR={win_rate:.1f}%")
    
    # 最大亏损交易
    print("\n4. Top 10 Worst Short Holds:")
    worst = df.nsmallest(10, 'pnl')
    for _, row in worst.iterrows():
        print(f"  {row['contract']} {row['open_time'][:10]}: {row['pnl']:.0f} pts, {row['hold_bars']:.0f} bars, entry@{row['entry_pct']:.2f}")
    
    # 统计
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total short holds: {len(df)}")
    print(f"Total PnL: {df['pnl'].sum():.0f} pts")
    print(f"Win rate: {df['is_winner'].mean()*100:.1f}%")
    print(f"Avg entry position: {df['entry_pct'].mean():.2f} (0=low, 1=high)")
    
    losers = df[~df['is_winner']]
    winners = df[df['is_winner']]
    print(f"\nLosers ({len(losers)}): avg entry={losers['entry_pct'].mean():.2f}")
    print(f"Winners ({len(winners)}): avg entry={winners['entry_pct'].mean():.2f}")
    
    # 保存
    output_path = ROOT / "experiments/iter17/short_hold_analysis.json"
    stats = {
        'total_short_holds': len(df),
        'total_pnl': float(df['pnl'].sum()),
        'win_rate': float(df['is_winner'].mean()),
        'avg_entry_pct': float(df['entry_pct'].mean()),
        'loser_avg_entry': float(losers['entry_pct'].mean()) if len(losers) > 0 else 0,
        'winner_avg_entry': float(winners['entry_pct'].mean()) if len(winners) > 0 else 0,
    }
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
