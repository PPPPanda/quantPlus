"""iter17: min_hold_bars 网格搜索."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import logging
import pandas as pd
from datetime import datetime
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

ROOT = Path(__file__).parent.parent

BENCHMARKS = [
    {"contract": "p2201.DCE", "csv": ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"},
    {"contract": "p2205.DCE", "csv": ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv"},
    {"contract": "p2305.DCE", "csv": ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv"},
    {"contract": "p2309.DCE", "csv": ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

BASE_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}


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


def run_7bench(setting):
    """跑 7 合约基准回测."""
    results = {}
    total_pnl = 0
    
    for bench in BENCHMARKS:
        vt_symbol = bench["contract"]
        csv_path = bench["csv"]
        
        start, end = import_csv_to_db(csv_path, vt_symbol)
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
        
        pnl = result.stats.get('total_net_pnl', 0) if result.stats else 0
        trades = result.stats.get('total_trade_count', 0) if result.stats else 0
        pnl_pts = pnl / 10  # 转点数
        
        results[vt_symbol] = {'pnl': pnl_pts, 'trades': trades}
        total_pnl += pnl_pts
    
    return results, total_pnl


def main():
    # min_hold_bars 网格
    mhb_values = [2, 3, 5, 7, 10]
    
    all_results = {}
    
    for mhb in mhb_values:
        print(f"\n{'='*60}")
        print(f"Testing min_hold_bars = {mhb}")
        print("="*60)
        
        setting = BASE_SETTING.copy()
        setting["min_hold_bars"] = mhb
        
        results, total = run_7bench(setting)
        
        all_results[f"mhb={mhb}"] = {
            'total': total,
            'by_contract': results
        }
        
        # 打印结果
        print(f"\nmin_hold_bars={mhb}: TOTAL={total:.1f} pts")
        for c, r in sorted(results.items(), key=lambda x: -x[1]['pnl']):
            status = "OK" if r['pnl'] >= -180 else "WARN" if r['pnl'] >= -500 else "FAIL"
            print(f"  {c}: {r['pnl']:+.1f} pts, {r['trades']} trades [{status}]")
    
    # 汇总对比
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    baseline_total = all_results.get("mhb=2", {}).get('total', 0)
    
    for config, data in sorted(all_results.items(), key=lambda x: -x[1]['total']):
        delta = data['total'] - baseline_total
        neg_contracts = [c for c, r in data['by_contract'].items() if r['pnl'] < -180]
        print(f"{config}: TOTAL={data['total']:.1f} (delta={delta:+.1f}) | neg<-180: {neg_contracts or 'None'}")
    
    # 保存结果
    output_path = ROOT / "experiments/iter17/mhb_grid_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
