"""
iter3 Phase 4 R1-improved: 混合背驰模式网格搜索

div_mode:
  0 = baseline (原始 diff 比较)
  1 = hybrid_or (diff OR divergence)
  2 = hybrid_filter (diff AND divergence; 面积不足时 fallback)
  3 = divergence_only (仅面积背驰)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from run_7bench import BENCHMARKS, BT_PARAMS, DEFAULT_SETTING, import_csv_to_db
from datetime import timedelta
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

# ---- Monkey-patch ----

_orig_on_5m = CtaChanPivotStrategy._on_5m_bar
_orig_process_bi = CtaChanPivotStrategy._process_bi
_orig_check_signal = CtaChanPivotStrategy._check_signal
_orig_init = CtaChanPivotStrategy.__init__


def _new_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._current_bi_macd_area = 0.0
    self._bi_macd_areas = []
    self._div_mode = getattr(CtaChanPivotStrategy, '_G_div_mode', 0)
    self._div_threshold = getattr(CtaChanPivotStrategy, '_G_div_threshold', 0.95)
    self._div_min_areas = getattr(CtaChanPivotStrategy, '_G_div_min_areas', 3)


def _new_on_5m(self, bar):
    _orig_on_5m(self, bar)
    histogram = self.diff_5m - self.dea_5m
    self._current_bi_macd_area += abs(histogram)


def _new_process_bi(self):
    old_len = len(self._bi_points)
    result = _orig_process_bi(self)
    if len(self._bi_points) > old_len:
        self._bi_macd_areas.append(self._current_bi_macd_area)
        self._current_bi_macd_area = 0.0
    return result


def _new_check_signal(self, curr_bar, new_bi):
    """_check_signal with divergence modes injected into 2B/2S logic."""
    if self._div_mode == 0:
        return _orig_check_signal(self, curr_bar, new_bi)

    # --- replicate original but with divergence in 2B/2S ---
    self._update_pivots()

    if len(self._bi_points) < 5:
        return

    # B02: cooldown
    if self._cooldown_remaining > 0:
        return

    p_now = self._bi_points[-1]
    p_last = self._bi_points[-2]
    p_prev = self._bi_points[-3]

    is_bull = self._prev_diff_15m > self._prev_dea_15m
    is_bear = self._prev_diff_15m < self._prev_dea_15m

    sig = None
    trigger_price = 0.0
    stop_base = 0.0
    last_pivot = self._pivots[-1] if self._pivots else None

    # 3B/3S (unchanged)
    if last_pivot:
        if p_now['type'] == 'bottom':
            if (p_now['price'] > last_pivot['zg'] and p_last['price'] > last_pivot['zg']
                and last_pivot['end_bi_idx'] >= len(self._bi_points) - self.pivot_valid_range
                and is_bull):
                sig = 'Buy'
                trigger_price = p_now['data']['high']
                stop_base = p_now['price']
        elif p_now['type'] == 'top':
            if (p_now['price'] < last_pivot['zd'] and p_last['price'] < last_pivot['zd']
                and last_pivot['end_bi_idx'] >= len(self._bi_points) - self.pivot_valid_range
                and is_bear):
                sig = 'CloseLong'
                trigger_price = p_now['data']['low']
                stop_base = p_now['price']

    # 2B/2S with divergence
    if not sig:
        n_areas = len(self._bi_macd_areas)

        def _has_divergence():
            if n_areas < self._div_min_areas:
                return None
            curr = self._bi_macd_areas[-1]
            # same-direction segment is 2 segments back (alternating up/down)
            if n_areas < 3:
                return None
            prev_same = self._bi_macd_areas[-3]
            if prev_same <= 0:
                return None
            return curr < prev_same * self._div_threshold

        if p_now['type'] == 'bottom':
            diff_ok = p_now['data']['diff'] > p_prev['data']['diff']
            price_ok = p_now['price'] > p_prev['price']
            div = _has_divergence()

            if self._div_mode == 1:
                cond = diff_ok or (div is True)
            elif self._div_mode == 2:
                cond = (diff_ok and div) if div is not None else diff_ok
            elif self._div_mode == 3:
                cond = div is True
            else:
                cond = diff_ok

            if price_ok and cond and is_bull:
                sig = 'Buy'
                trigger_price = p_now['data']['high']
                stop_base = p_now['price']

        elif p_now['type'] == 'top':
            diff_ok = p_now['data']['diff'] < p_prev['data']['diff']
            price_ok = p_now['price'] < p_prev['price']
            div = _has_divergence()

            if self._div_mode == 1:
                cond = diff_ok or (div is True)
            elif self._div_mode == 2:
                cond = (diff_ok and div) if div is not None else diff_ok
            elif self._div_mode == 3:
                cond = div is True
            else:
                cond = diff_ok

            if price_ok and cond and is_bear:
                sig = 'CloseLong'
                trigger_price = p_now['data']['low']
                stop_base = p_now['price']

    # Signal filtering (replicate original)
    if sig and self.atr > 0:
        if sig == 'CloseLong':
            if self._position == 1:
                self._signal_type = ("3S" if (self._pivots and last_pivot and
                    p_now['price'] < last_pivot['zd']) else "2S")
                self._pending_signal = {
                    'type': 'CloseLong',
                    'trigger_price': trigger_price,
                    'stop_base': stop_base
                }
                self.signal = f"待平多({self._signal_type})"
            return

        distance = abs(trigger_price - stop_base)
        if distance < self.atr_entry_filter * self.atr:
            if self._pivots and last_pivot:
                if p_now['type'] == 'bottom' and p_now['price'] > last_pivot['zg']:
                    self._signal_type = "3B"
                else:
                    self._signal_type = "2B"
            else:
                self._signal_type = "2B"
            self._pending_signal = {
                'type': sig,
                'trigger_price': trigger_price,
                'stop_base': stop_base
            }
            self.signal = f"待{sig}({self._signal_type})"


CtaChanPivotStrategy.__init__ = _new_init
CtaChanPivotStrategy._on_5m_bar = _new_on_5m
CtaChanPivotStrategy._process_bi = _new_process_bi
CtaChanPivotStrategy._check_signal = _new_check_signal


# ---- Grid ----
def gen_combos():
    combos = [{"div_mode": 0, "div_threshold": 0, "min_areas": 0}]  # baseline
    for mode in [1, 2, 3]:
        for thresh in [0.70, 0.80, 0.90, 0.95, 1.0]:
            for ma in [2, 3]:
                combos.append({"div_mode": mode, "div_threshold": thresh, "min_areas": ma})
    return combos


def run_combo(combo):
    CtaChanPivotStrategy._G_div_mode = combo["div_mode"]
    CtaChanPivotStrategy._G_div_threshold = combo["div_threshold"]
    CtaChanPivotStrategy._G_div_min_areas = combo["min_areas"]

    setting = dict(DEFAULT_SETTING)
    results = []
    for bench in BENCHMARKS:
        vt = bench["contract"]
        start, end, _ = import_csv_to_db(bench["csv"], vt)
        r = run_backtest(
            vt_symbol=vt, start=start - timedelta(days=1), end=end + timedelta(days=1),
            strategy_class=CtaChanPivotStrategy, strategy_setting=setting, **BT_PARAMS,
        )
        s = r.stats or {}
        results.append({
            "contract": vt.split(".")[0], "slot": bench["slot"],
            "pnl": round(s.get("total_net_pnl", 0), 1),
            "trades": s.get("total_trade_count", 0),
            "sharpe": round(s.get("sharpe_ratio", 0), 2),
        })
    return results


def main():
    combos = gen_combos()
    print(f"Total combos: {len(combos)}  (x7 contracts = {len(combos)*7} backtests)")
    print("=" * 130)

    all_results = []
    for i, combo in enumerate(combos):
        label = f"m={combo['div_mode']} t={combo['div_threshold']:.2f} a={combo['min_areas']}"
        print(f"[{i+1}/{len(combos)}] {label}...", end=" ", flush=True)
        t0 = time.time()
        results = run_combo(combo)
        elapsed = time.time() - t0

        total = sum(r["pnl"] for r in results)
        pts = total / 10.0
        neg = [r["contract"] for r in results if r["pnl"] < 0]
        status = "PASS" if pts >= 5600 else "FAIL"

        row = {"combo": combo, "results": results, "total_pnl": total,
               "total_pts": pts, "neg_contracts": neg, "n_neg": len(neg), "status": status}
        all_results.append(row)

        per_c = " ".join(f"{r['contract']}={r['pnl']:>+7.0f}({r['trades']:>3d})" for r in results)
        print(f"pts={pts:>7.1f} neg={len(neg)} {status} [{elapsed:.0f}s] | {per_c}")

    all_results.sort(key=lambda x: (x["n_neg"], -x["total_pts"]))

    print("\n" + "=" * 130)
    print("TOP 15 (sorted: fewest negatives first, then highest total):")
    for i, row in enumerate(all_results[:15]):
        c = row["combo"]
        neg_detail = ", ".join(
            f"{n}={next(r['pnl'] for r in row['results'] if r['contract']==n):.0f}"
            for n in row["neg_contracts"]
        )
        per_c = " ".join(f"{r['contract']}={r['pnl']:>+7.0f}" for r in row['results'])
        print(f"  #{i+1}: m={c['div_mode']} t={c['div_threshold']:.2f} a={c['min_areas']} | "
              f"pts={row['total_pts']:>7.1f} neg={row['n_neg']} {row['status']} | {per_c}")

    out = ROOT / "experiments/iter3/p4_rounds/r1_improved_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
