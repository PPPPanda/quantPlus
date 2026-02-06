"""
分析跳空后交易的表现 - 对比 p2209, p2601 vs p2401
"""
import sys
sys.path.insert(0, '/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/src')

import json
from datetime import datetime, timedelta
from pathlib import Path
from qp.core import FractalMaestroStrategy
from qp.data.loader import load_bars_wind, load_bars_xt

def analyze_contract_gap_trades(contract: str, gap_data: list):
    """分析合约的跳空后交易表现"""
    # 加载数据
    if contract == 'p2601':
        bars = load_bars_xt(contract, '1m')
    else:
        bars = load_bars_wind(contract, '1m')
    
    if bars.empty:
        return None
    
    # 运行策略
    strategy = FractalMaestroStrategy(
        debug_enabled=False,
        debug_log_console=False,
        cooldown_losses=2,
        cooldown_bars=20,
        atr_activate_mult=2.5,
        atr_trailing_mult=3.0,
        atr_entry_filter=2.0
    )
    
    for _, bar in bars.iterrows():
        strategy.on_bar(bar)
    
    trades = strategy.closed_trades
    
    # 分析跳空后的交易
    results = []
    for gap in gap_data:
        gap_dt = datetime.strptime(gap['datetime'], '%Y-%m-%d %H:%M:%S')
        gap_date = gap_dt.date()
        in_fractal = gap['gap_in_fractal']
        
        # 找跳空后24小时内的交易
        post_gap_trades = []
        for t in trades:
            trade_dt = t['entry_time']
            if isinstance(trade_dt, str):
                trade_dt = datetime.strptime(trade_dt, '%Y-%m-%d %H:%M:%S')
            
            if gap_dt <= trade_dt <= gap_dt + timedelta(hours=24):
                post_gap_trades.append(t)
        
        # 统计这些交易的盈亏
        total_pnl = sum(t['pnl'] for t in post_gap_trades)
        num_trades = len(post_gap_trades)
        
        results.append({
            'date': gap_dt.strftime('%Y-%m-%d'),
            'gap': gap['gap'],
            'gap_atr': gap['gap_atr'],
            'in_fractal': in_fractal,
            'post_gap_trades': num_trades,
            'post_gap_pnl': total_pnl,
            'trades': post_gap_trades
        })
    
    return results

def main():
    # 加载跳空数据
    with open('experiments/iter18_gap_fractal/gap_fractal_impact.json') as f:
        data = json.load(f)
    
    contracts = ['p2209', 'p2601', 'p2401']
    
    for contract in contracts:
        print(f'\n{"="*60}')
        print(f'{contract} 跳空后交易分析')
        print(f'{"="*60}')
        
        gap_data = data['by_contract'][contract]['gaps']
        results = analyze_contract_gap_trades(contract, gap_data)
        
        if not results:
            print(f'  无法加载数据')
            continue
        
        # 统计
        in_fractal_pnl = 0
        out_fractal_pnl = 0
        in_fractal_trades = 0
        out_fractal_trades = 0
        
        print(f'\n日期          | 跳空   | ATR  | 分型? | 交易数 | 盈亏')
        print(f'-'*70)
        
        for r in results:
            status = '🔴' if r['in_fractal'] else '🟢'
            print(f"{r['date']} | {r['gap']:+6.0f} | {r['gap_atr']:4.1f} | {status}    | {r['post_gap_trades']:4d}   | {r['post_gap_pnl']:+.0f}")
            
            if r['in_fractal']:
                in_fractal_pnl += r['post_gap_pnl']
                in_fractal_trades += r['post_gap_trades']
            else:
                out_fractal_pnl += r['post_gap_pnl']
                out_fractal_trades += r['post_gap_trades']
        
        print(f'\n汇总:')
        print(f'  分型内跳空后: {in_fractal_trades} 笔交易, 总盈亏 {in_fractal_pnl:+.0f}')
        print(f'  分型外跳空后: {out_fractal_trades} 笔交易, 总盈亏 {out_fractal_pnl:+.0f}')

if __name__ == '__main__':
    main()
