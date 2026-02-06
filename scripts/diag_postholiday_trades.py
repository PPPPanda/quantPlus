"""
诊断：节后开盘信号的实际胜率分析

目标：
1. 识别节后开盘的交易（节假日后第一个交易日的信号）
2. 统计节后交易 vs 普通交易的胜率/盈亏
3. 量化"节后信号质量下降"的程度
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pathlib import Path
import json

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Direction, Offset

# 复用 run_7bench 的辅助函数
from scripts.run_7bench import import_csv_to_db

# 数据路径
DATA_DIR = Path(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\data\analyse")

# 合约映射
CONTRACTS = {
    'p2201': ('p2201_1min_202108-202112.csv', '2021-08-01', '2021-12-31'),
    'p2205': ('p2205_1min_202112-202204.csv', '2021-12-01', '2022-04-30'),
    'p2209': ('p2209_1min_202204-202208.csv', '2022-04-01', '2022-08-31'),
    'p2301': ('p2301_1min_202208-202212.csv', '2022-08-01', '2022-12-31'),
    'p2305': ('p2305_1min_202212-202304.csv', '2022-12-01', '2023-04-30'),
    'p2309': ('p2309_1min_202304-202308.csv', '2023-04-01', '2023-08-31'),
    'p2401': ('p2401_1min_202308-202312.csv', '2023-08-01', '2023-12-31'),
}

# 节假日列表（用于判断节后）
HOLIDAYS = {
    # 2021
    "2021-10-01": "国庆", "2021-10-02": "国庆", "2021-10-03": "国庆",
    "2021-10-04": "国庆", "2021-10-05": "国庆", "2021-10-06": "国庆", "2021-10-07": "国庆",
    # 2022
    "2022-01-31": "春节", "2022-02-01": "春节", "2022-02-02": "春节",
    "2022-02-03": "春节", "2022-02-04": "春节",
    "2022-04-04": "清明", "2022-04-05": "清明",
    "2022-05-02": "劳动节", "2022-05-03": "劳动节", "2022-05-04": "劳动节",
    "2022-06-03": "端午",
    "2022-09-12": "中秋",
    "2022-10-03": "国庆", "2022-10-04": "国庆", "2022-10-05": "国庆",
    "2022-10-06": "国庆", "2022-10-07": "国庆",
    # 2023
    "2023-01-23": "春节", "2023-01-24": "春节", "2023-01-25": "春节",
    "2023-01-26": "春节", "2023-01-27": "春节",
    "2023-04-05": "清明",
    "2023-05-01": "劳动节", "2023-05-02": "劳动节", "2023-05-03": "劳动节",
    "2023-06-22": "端午", "2023-06-23": "端午",
    "2023-09-29": "中秋+国庆",
    "2023-10-02": "国庆", "2023-10-03": "国庆", "2023-10-04": "国庆",
    "2023-10-05": "国庆", "2023-10-06": "国庆",
}

# 节后首日集合（预计算）
POST_HOLIDAY_DATES = set()
def _init_post_holiday_dates():
    """预计算所有节后第一个交易日"""
    for date_str in HOLIDAYS.keys():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # 检查后面的日期，找到第一个非节假日、非周末的日期
        for days_ahead in range(1, 10):
            next_dt = dt + timedelta(days=days_ahead)
            next_str = next_dt.strftime("%Y-%m-%d")
            if next_str not in HOLIDAYS and next_dt.weekday() < 5:
                POST_HOLIDAY_DATES.add(next_str)
                break
_init_post_holiday_dates()


def is_post_holiday(trade_date: str) -> tuple:
    """判断某日期是否是节后第一个交易日"""
    if trade_date in POST_HOLIDAY_DATES:
        # 找到对应的节假日名称
        dt = datetime.strptime(trade_date, "%Y-%m-%d")
        for days_back in range(1, 10):
            check_dt = dt - timedelta(days=days_back)
            check_str = check_dt.strftime("%Y-%m-%d")
            if check_str in HOLIDAYS:
                return True, HOLIDAYS[check_str]
        return True, "unknown"
    return False, ""


def run_backtest_and_pair_trades(contract: str, csv_file: str, start: str, end: str):
    """运行回测并配对交易计算 PnL"""
    from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
    
    vt_symbol = f"{contract}.DCE"
    csv_path = DATA_DIR / csv_file
    
    # 导入数据
    import_csv_to_db(str(csv_path), vt_symbol)
    
    # 创建回测引擎
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval="1m",
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        rate=1e-4,
        slippage=1,
        size=10,
        pricetick=2,
        capital=1_000_000,
    )
    
    # 策略参数（当前最优）
    setting = {
        "circuit_breaker_losses": 7,
        "circuit_breaker_bars": 70,
        "div_threshold": 0.39,
        "max_pullback_atr": 3.2,
    }
    
    engine.add_strategy(CtaChanPivotStrategy, setting)
    engine.load_data()
    engine.run_backtesting()
    
    # 获取原始交易记录
    raw_trades = engine.get_all_trades()
    
    # 配对交易计算 PnL
    paired_trades = []
    open_positions = []  # 存储未平仓的开仓记录
    
    for trade in raw_trades:
        if trade.offset == Offset.OPEN:
            # 开仓记录
            open_positions.append({
                'entry_time': trade.datetime,
                'entry_price': trade.price,
                'direction': trade.direction,
                'volume': trade.volume,
            })
        elif trade.offset == Offset.CLOSE and open_positions:
            # 平仓记录 - 配对最早的开仓（FIFO）
            open_trade = open_positions.pop(0)
            
            # 计算 PnL（点数，已乘以 size=10）
            if open_trade['direction'] == Direction.LONG:
                pnl_points = (trade.price - open_trade['entry_price'])
            else:
                pnl_points = (open_trade['entry_price'] - trade.price)
            
            # 扣除滑点和手续费（近似）
            # slippage = 1 * 2 = 2 pts (开+平各1)
            # commission ≈ price * 1e-4 * size ≈ 9000 * 0.0001 * 10 ≈ 9 pts (约 0.1%)
            pnl_net = pnl_points - 2 - 1.8  # 近似扣除
            
            paired_trades.append({
                'contract': contract,
                'entry_time': open_trade['entry_time'],
                'exit_time': trade.datetime,
                'entry_price': open_trade['entry_price'],
                'exit_price': trade.price,
                'direction': 'LONG' if open_trade['direction'] == Direction.LONG else 'SHORT',
                'pnl': pnl_net,
                'holding_bars': None,  # 后续可计算
            })
    
    return paired_trades


def analyze_trades(all_trades: list):
    """分析交易的节后 vs 普通表现"""
    post_holiday_trades = []
    normal_trades = []
    
    for trade in all_trades:
        trade_date = trade['entry_time'].strftime("%Y-%m-%d")
        is_ph, holiday_name = is_post_holiday(trade_date)
        
        trade_info = {
            'date': trade_date,
            'entry_time': str(trade['entry_time']),
            'exit_time': str(trade['exit_time']),
            'pnl': trade['pnl'],
            'is_winner': trade['pnl'] > 0,
            'contract': trade['contract'],
            'direction': trade['direction'],
        }
        
        if is_ph:
            trade_info['holiday'] = holiday_name
            post_holiday_trades.append(trade_info)
        else:
            normal_trades.append(trade_info)
    
    return post_holiday_trades, normal_trades


def main():
    print("=" * 60)
    print("节后开盘信号的实际胜率分析")
    print("=" * 60)
    
    all_trades = []
    
    for contract, (csv_file, start, end) in CONTRACTS.items():
        print(f"\n--- {contract} ---")
        csv_path = DATA_DIR / csv_file
        if not csv_path.exists():
            print(f"  [SKIP] 文件不存在")
            continue
        
        try:
            trades = run_backtest_and_pair_trades(contract, csv_file, start, end)
            print(f"  配对交易数: {len(trades)}")
            all_trades.extend(trades)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    if not all_trades:
        print("\n没有交易数据，无法分析")
        return
    
    # 分析
    post_holiday, normal = analyze_trades(all_trades)
    
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    
    # 节后交易统计
    ph_wins = 0
    ph_total_pnl = 0
    ph_win_rate = 0
    if post_holiday:
        ph_wins = sum(1 for t in post_holiday if t['is_winner'])
        ph_total_pnl = sum(t['pnl'] for t in post_holiday)
        ph_win_rate = ph_wins / len(post_holiday) * 100
        print(f"\n节后交易:")
        print(f"  数量: {len(post_holiday)}")
        print(f"  胜率: {ph_win_rate:.1f}%")
        print(f"  总盈亏: {ph_total_pnl:.1f} pts")
        print(f"  平均盈亏: {ph_total_pnl/len(post_holiday):.1f} pts")
        print(f"  样本:")
        for t in post_holiday[:5]:
            print(f"    {t['date']} ({t['holiday']}): {t['pnl']:+.1f} pts")
    else:
        print("\n节后交易: 0")
    
    # 普通交易统计
    n_wins = 0
    n_total_pnl = 0
    n_win_rate = 0
    if normal:
        n_wins = sum(1 for t in normal if t['is_winner'])
        n_total_pnl = sum(t['pnl'] for t in normal)
        n_win_rate = n_wins / len(normal) * 100
        print(f"\n普通交易:")
        print(f"  数量: {len(normal)}")
        print(f"  胜率: {n_win_rate:.1f}%")
        print(f"  总盈亏: {n_total_pnl:.1f} pts")
        print(f"  平均盈亏: {n_total_pnl/len(normal):.1f} pts")
    
    # 对比
    if post_holiday and normal:
        print(f"\n胜率差异: {ph_win_rate - n_win_rate:+.1f}%")
        print(f"  (负数表示节后信号质量更差)")
        print(f"平均盈亏差异: {ph_total_pnl/len(post_holiday) - n_total_pnl/len(normal):+.1f} pts")
    
    # 保存详细结果
    output = {
        'summary': {
            'total_trades': len(all_trades),
            'post_holiday_count': len(post_holiday),
            'normal_count': len(normal),
        },
        'post_holiday': {
            'count': len(post_holiday),
            'win_rate': ph_win_rate,
            'total_pnl': ph_total_pnl,
            'avg_pnl': ph_total_pnl / len(post_holiday) if post_holiday else 0,
            'trades': post_holiday,  # 保存全部节后交易（数量较少）
        },
        'normal': {
            'count': len(normal),
            'win_rate': n_win_rate,
            'total_pnl': n_total_pnl,
            'avg_pnl': n_total_pnl / len(normal) if normal else 0,
        },
        'comparison': {
            'win_rate_diff': ph_win_rate - n_win_rate if (post_holiday and normal) else None,
            'avg_pnl_diff': (ph_total_pnl/len(post_holiday) - n_total_pnl/len(normal)) if (post_holiday and normal) else None,
        }
    }
    
    output_path = Path(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\experiments\iter17\postholiday_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
