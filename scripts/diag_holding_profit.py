"""诊断脚本：分析持仓时长与收益的关系."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

ROOT = Path(__file__).parent.parent

BENCHMARKS = [
    ("p2201.DCE", ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"),
    ("p2205.DCE", ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv"),
    ("p2209.DCE", ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"),
    ("p2301.DCE", ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv"),
    ("p2305.DCE", ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv"),
    ("p2309.DCE", ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv"),
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
    return df["datetime"].min(), df["datetime"].max()


def run_with_trade_log(vt_symbol, csv_path, setting):
    """跑回测并返回交易记录."""
    start, end = import_csv_to_db(csv_path, vt_symbol)
    # 转换为 Python datetime
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
    return result


def analyze_trades(result):
    """分析交易记录，提取持仓时长和收益."""
    trades = result.trades  # list of TradeData
    if not trades:
        return []
    
    # 配对交易（开仓-平仓）
    paired = []
    open_trade = None
    
    for t in trades:
        if t.offset.value == "开":  # 开仓
            open_trade = t
        elif t.offset.value == "平" and open_trade:  # 平仓
            # 计算持仓时长（分钟）
            hold_minutes = (t.datetime - open_trade.datetime).total_seconds() / 60
            # 计算收益（点数）
            if open_trade.direction.value == "多":
                pnl = (t.price - open_trade.price)
            else:
                pnl = (open_trade.price - t.price)
            
            paired.append({
                'open_time': open_trade.datetime,
                'close_time': t.datetime,
                'direction': open_trade.direction.value,
                'hold_minutes': hold_minutes,
                'hold_bars_5m': hold_minutes / 5,
                'pnl': pnl,
                'open_price': open_trade.price,
                'close_price': t.price,
            })
            open_trade = None
    
    return paired


def main():
    setting = {
        "debug_enabled": False,
        "debug_log_console": False,
        # 使用当前最优参数
        "circuit_breaker_losses": 7,
        "circuit_breaker_bars": 70,
        "div_threshold": 0.39,
        "max_pullback_atr": 3.2,
    }
    
    all_trades = []
    
    for contract, csv_path in BENCHMARKS:
        print(f"Processing {contract}...", flush=True)
        result = run_with_trade_log(contract, csv_path, setting)
        trades = analyze_trades(result)
        for t in trades:
            t['contract'] = contract
        all_trades.extend(trades)
        pnl = result.stats.get('total_net_pnl', 0) if result.stats else 0
        print(f"  {len(trades)} trades, PnL={pnl:.0f}")
    
    if not all_trades:
        print("No trades found!")
        return
    
    df = pd.DataFrame(all_trades)
    
    print("\n" + "="*60)
    print("HOLDING TIME vs PROFIT ANALYSIS")
    print("="*60)
    
    # 按持仓时长分组
    bins = [0, 5, 10, 20, 50, 100, 200, 500, float('inf')]
    labels = ['<5', '5-10', '10-20', '20-50', '50-100', '100-200', '200-500', '>500']
    df['hold_group'] = pd.cut(df['hold_bars_5m'], bins=bins, labels=labels)
    
    grouped = df.groupby('hold_group').agg({
        'pnl': ['count', 'sum', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    })
    grouped.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']
    
    print("\nBy Holding Time (5m bars):")
    print(grouped.to_string())
    
    # 盈利交易 vs 亏损交易的持仓时长
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    print(f"\n\nWinners ({len(winners)} trades):")
    print(f"  Avg holding: {winners['hold_bars_5m'].mean():.1f} bars")
    print(f"  Avg profit: {winners['pnl'].mean():.1f} pts")
    print(f"  Max profit: {winners['pnl'].max():.1f} pts")
    
    print(f"\nLosers ({len(losers)} trades):")
    print(f"  Avg holding: {losers['hold_bars_5m'].mean():.1f} bars")
    print(f"  Avg loss: {losers['pnl'].mean():.1f} pts")
    print(f"  Max loss: {losers['pnl'].min():.1f} pts")
    
    # 长持仓（>50 bars）的收益分布
    long_holds = df[df['hold_bars_5m'] > 50]
    short_holds = df[df['hold_bars_5m'] <= 50]
    
    print(f"\n\nLong holds (>50 bars): {len(long_holds)} trades")
    print(f"  Total PnL: {long_holds['pnl'].sum():.1f} pts")
    print(f"  Win rate: {(long_holds['pnl'] > 0).mean() * 100:.1f}%")
    
    print(f"\nShort holds (<=50 bars): {len(short_holds)} trades")
    print(f"  Total PnL: {short_holds['pnl'].sum():.1f} pts")
    print(f"  Win rate: {(short_holds['pnl'] > 0).mean() * 100:.1f}%")
    
    # 超长持仓（>200 bars，约 1 天）
    very_long = df[df['hold_bars_5m'] > 200]
    if len(very_long) > 0:
        print(f"\n\nVery long holds (>200 bars, ~1 day): {len(very_long)} trades")
        print(f"  Total PnL: {very_long['pnl'].sum():.1f} pts")
        print(f"  Avg PnL: {very_long['pnl'].mean():.1f} pts")
        print(f"  Win rate: {(very_long['pnl'] > 0).mean() * 100:.1f}%")
    
    # 保存详细数据
    output_path = ROOT / "experiments/iter17/trade_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    
    stats = {
        'total_trades': len(df),
        'total_pnl': float(df['pnl'].sum()),
        'avg_hold_bars': float(df['hold_bars_5m'].mean()),
        'winner_avg_hold': float(winners['hold_bars_5m'].mean()),
        'loser_avg_hold': float(losers['hold_bars_5m'].mean()),
        'long_hold_pnl': float(long_holds['pnl'].sum()),
        'short_hold_pnl': float(short_holds['pnl'].sum()),
        'holding_distribution': grouped.reset_index().to_dict('records'),
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
