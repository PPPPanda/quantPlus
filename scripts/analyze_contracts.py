"""
iter6 Phase 1: 合约统计特征分析.

深入分析每个合约的:
1. 行情统计（涨跌幅、波动率、趋势性）
2. 策略交易统计（逐笔、按月、按信号类型）
3. 失败模式定位（连亏段、最差交易的行情环境）

用法:
    cd quantPlus
    .venv/Scripts/python.exe scripts/analyze_contracts.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

BENCHMARKS = [
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv", "source": "Wind"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
}


def analyze_market_stats(df: pd.DataFrame, contract: str) -> dict:
    """分析行情统计特征."""
    # 日线聚合
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index()

    # 基本统计
    first_close = daily['close'].iloc[0]
    last_close = daily['close'].iloc[-1]
    total_return = (last_close - first_close) / first_close * 100
    high_price = daily['high'].max()
    low_price = daily['low'].min()
    max_range = (high_price - low_price) / first_close * 100

    # 日收益率
    daily['ret'] = daily['close'].pct_change()
    daily_vol = daily['ret'].std() * 100
    ann_vol = daily_vol * np.sqrt(252)

    # 趋势性指标
    daily['direction'] = np.sign(daily['ret'])
    up_days = (daily['direction'] > 0).sum()
    down_days = (daily['direction'] < 0).sum()
    total_days = len(daily)

    # 连续涨跌统计
    streaks = []
    current_dir = 0
    current_len = 0
    for d in daily['direction'].dropna():
        if d == current_dir:
            current_len += 1
        else:
            if current_len > 0:
                streaks.append((current_dir, current_len))
            current_dir = d
            current_len = 1
    if current_len > 0:
        streaks.append((current_dir, current_len))

    max_up_streak = max((l for d, l in streaks if d > 0), default=0)
    max_down_streak = max((l for d, l in streaks if d < 0), default=0)

    # 月度收益
    daily['month'] = pd.to_datetime(daily['date']).dt.to_period('M')
    monthly = daily.groupby('month').agg(
        open=('open', 'first'),
        close=('close', 'last'),
    )
    monthly['ret'] = (monthly['close'] - monthly['open']) / monthly['open'] * 100

    # 5m ATR 近似（用日高低差）
    daily['range'] = daily['high'] - daily['low']
    avg_daily_range = daily['range'].mean()
    avg_daily_range_pct = (avg_daily_range / daily['close'].mean()) * 100

    return {
        'contract': contract,
        'trading_days': total_days,
        'total_return_pct': round(total_return, 2),
        'max_range_pct': round(max_range, 2),
        'daily_vol_pct': round(daily_vol, 3),
        'ann_vol_pct': round(ann_vol, 1),
        'up_days': int(up_days),
        'down_days': int(down_days),
        'up_ratio': round(up_days / max(1, up_days + down_days) * 100, 1),
        'max_up_streak': int(max_up_streak),
        'max_down_streak': int(max_down_streak),
        'avg_daily_range_pct': round(avg_daily_range_pct, 2),
        'price_range': f"{low_price:.0f}-{high_price:.0f}",
        'monthly_returns': {str(k): round(v, 2) for k, v in monthly['ret'].items()},
    }


def analyze_trades(result, contract: str) -> dict:
    """分析交易统计."""
    trades = getattr(result, 'trades', None) or getattr(result, 'all_trades', [])
    if not trades:
        return {'contract': contract, 'error': 'no trades'}

    # 构建 round-trip
    round_trips = []
    open_trade = None
    for t in trades:
        if t.offset.value == '开':
            open_trade = t
        elif t.offset.value == '平' and open_trade:
            pnl_pts = (t.price - open_trade.price) if open_trade.direction.value == '多' else (open_trade.price - t.price)
            duration = (t.datetime - open_trade.datetime).total_seconds() / 60
            round_trips.append({
                'open_time': str(open_trade.datetime),
                'close_time': str(t.datetime),
                'direction': open_trade.direction.value,
                'open_price': open_trade.price,
                'close_price': t.price,
                'pnl_pts': pnl_pts,
                'duration_min': duration,
                'month': open_trade.datetime.strftime('%Y-%m'),
            })
            open_trade = None

    if not round_trips:
        return {'contract': contract, 'error': 'no round trips'}

    wins = [r for r in round_trips if r['pnl_pts'] > 0]
    losses = [r for r in round_trips if r['pnl_pts'] <= 0]

    # 月度分解
    monthly_pnl = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0})
    for r in round_trips:
        m = r['month']
        monthly_pnl[m]['count'] += 1
        monthly_pnl[m]['pnl'] += r['pnl_pts']
        if r['pnl_pts'] > 0:
            monthly_pnl[m]['wins'] += 1
        else:
            monthly_pnl[m]['losses'] += 1

    # 持仓时长分布
    durations = [r['duration_min'] for r in round_trips]
    short_trades = sum(1 for d in durations if d < 30)  # <30min
    medium_trades = sum(1 for d in durations if 30 <= d < 240)  # 30min-4h
    long_trades = sum(1 for d in durations if d >= 240)  # >4h

    # 连亏分析
    streaks = []
    current_streak = 0
    current_pnl = 0
    for r in round_trips:
        if r['pnl_pts'] <= 0:
            current_streak += 1
            current_pnl += r['pnl_pts']
        else:
            if current_streak > 0:
                streaks.append({'length': current_streak, 'pnl': round(current_pnl, 1)})
            current_streak = 0
            current_pnl = 0
    if current_streak > 0:
        streaks.append({'length': current_streak, 'pnl': round(current_pnl, 1)})
    worst_streaks = sorted(streaks, key=lambda x: x['pnl'])[:3]

    return {
        'contract': contract,
        'total_trades': len(round_trips),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(round_trips) * 100, 1),
        'avg_win': round(sum(r['pnl_pts'] for r in wins) / max(1, len(wins)), 1),
        'avg_loss': round(sum(r['pnl_pts'] for r in losses) / max(1, len(losses)), 1),
        'total_pnl': round(sum(r['pnl_pts'] for r in round_trips), 1),
        'profit_factor': round(
            sum(r['pnl_pts'] for r in wins) / max(0.01, abs(sum(r['pnl_pts'] for r in losses))),
            2
        ),
        'avg_duration_min': round(np.mean(durations), 0),
        'short_trades_pct': round(short_trades / len(round_trips) * 100, 1),
        'medium_trades_pct': round(medium_trades / len(round_trips) * 100, 1),
        'long_trades_pct': round(long_trades / len(round_trips) * 100, 1),
        'monthly_pnl': {k: {'count': v['count'], 'pnl': round(v['pnl'], 1), 'wr': round(v['wins']/max(1,v['count'])*100, 0)}
                       for k, v in sorted(monthly_pnl.items())},
        'worst_streaks': worst_streaks,
    }


def get_database():
    """复用 run_7bench 的数据库获取方式."""
    from peewee import SqliteDatabase
    from vnpy.trader.database import get_database as vnpy_get_db
    db = vnpy_get_db()
    return db


def main():
    from qp.backtest.engine import run_backtest
    db = get_database()

    all_market = []
    all_trades = []

    for bench in BENCHMARKS:
        contract = bench['contract']
        csv_path = bench['csv']
        short = contract.split('.')[0]
        print(f"  {short}...", end=" ", flush=True)

        # 读取并分析行情
        df = pd.read_csv(csv_path)
        df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)
        market = analyze_market_stats(df_norm, short)
        all_market.append(market)

        # 回测
        start = pd.Timestamp(df_norm['datetime'].min()).to_pydatetime()
        end = pd.Timestamp(df_norm['datetime'].max()).to_pydatetime()
        result = run_backtest(
            vt_symbol=contract,
            interval=BT_PARAMS['interval'],
            start=start, end=end,
            rate=BT_PARAMS['rate'],
            slippage=BT_PARAMS['slippage'],
            size=BT_PARAMS['size'],
            pricetick=BT_PARAMS['pricetick'],
            capital=BT_PARAMS['capital'],
            strategy_class=CtaChanPivotStrategy,
            strategy_setting=DEFAULT_SETTING,
        )

        # 分析交易
        trade_stats = analyze_trades(result, short)
        all_trades.append(trade_stats)

        # 获取 stats
        stats = result.stats if hasattr(result, 'stats') else result
        total_net = stats.get('total_net_pnl', stats.get('total_pnl', 0))
        pts = total_net / 10
        trades_count = stats.get('total_trade_count', 0)
        print(f"pts={pts:>8.1f}  trades={trades_count:>4d}")

    # 保存结果
    output = {
        'market_stats': all_market,
        'trade_stats': all_trades,
    }

    out_path = ROOT / "experiments" / "iter6" / "contract_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
