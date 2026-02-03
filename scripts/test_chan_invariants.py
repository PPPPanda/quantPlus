"""test_chan_invariants.py

缠论结构不变量测试：
1. 分型交替（顶↔底）
2. 笔端点方向一致性
3. 中枢区间 zg > zd
4. 笔间距 >= min_bi_gap
5. 中枢重叠/递进检查
6. 包含处理后K线不应存在包含关系

在 7 个合约数据上运行，输出违规统计。
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math


class ChanStructureExtractor:
    """Extract chan structure (k_lines, bi_points, pivots) from 1min data."""

    def __init__(self, df_1m: pd.DataFrame, min_bi_gap: int = 4):
        self.min_bi_gap = min_bi_gap
        self.df_1m = df_1m.reset_index(drop=True)

        df_idx = self.df_1m.set_index("datetime")
        df_idx.index = pd.to_datetime(df_idx.index)
        self.df_5m = (
            df_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

        # State
        self.k_lines: List[Dict] = []
        self.inclusion_dir: int = 0
        self.bi_points: List[Dict] = []
        self.pivots: List[Dict] = []

        # Violations
        self.violations: List[Dict] = []

        self._build()

    def _build(self):
        for ts, row in self.df_5m.iterrows():
            bar = {"high": float(row["high"]), "low": float(row["low"]),
                   "time": ts, "diff": 0.0, "atr": 0.0,
                   "diff_15m": 0.0, "dea_15m": 0.0}
            self._inclusion(bar)
            self._bi()
            self._pivots_update()

    def _inclusion(self, nb):
        if not self.k_lines:
            self.k_lines.append(nb)
            return

        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]

        if il or inw:
            if self.inclusion_dir == 0:
                self.inclusion_dir = 1
            m = last.copy()
            m["time"] = nb["time"]
            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"])
                m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"])
                m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]:
                self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]:
                self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self):
        if len(self.k_lines) < 3:
            return
        c, m2, l = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(self.k_lines) - 2, "data": m2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(self.k_lines) - 2, "data": m2}

        if not cand:
            return
        if not self.bi_points:
            self.bi_points.append(cand)
            return

        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]:
                self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]:
                self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= self.min_bi_gap:
                self.bi_points.append(cand)

    def _pivots_update(self):
        if len(self.bi_points) < 4:
            return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i + 1]["price"]),
                    max(pts[i]["price"], pts[i + 1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges)
        zd = max(r[0] for r in ranges)
        if zg > zd:
            new_p = {"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points) - 1,
                     "start_bi_idx": len(self.bi_points) - 4}
            # Avoid duplicating identical pivots
            if not self.pivots or self.pivots[-1]["end_bi_idx"] != new_p["end_bi_idx"]:
                self.pivots.append(new_p)

    def check_invariants(self, contract: str) -> List[Dict]:
        violations = []

        # 1. Bi alternation: top ↔ bottom
        for i in range(1, len(self.bi_points)):
            if self.bi_points[i]["type"] == self.bi_points[i - 1]["type"]:
                violations.append({
                    "contract": contract,
                    "rule": "bi_alternation",
                    "detail": f"bi[{i-1}] and bi[{i}] both {self.bi_points[i]['type']}",
                    "severity": "HIGH"
                })

        # 2. Bi gap >= min_bi_gap
        for i in range(1, len(self.bi_points)):
            gap = self.bi_points[i]["idx"] - self.bi_points[i - 1]["idx"]
            if gap < self.min_bi_gap:
                violations.append({
                    "contract": contract,
                    "rule": "bi_min_gap",
                    "detail": f"bi[{i-1}]→bi[{i}] gap={gap} < {self.min_bi_gap}",
                    "severity": "HIGH"
                })

        # 3. Bi direction consistency (top.price > bottom.price for each bi)
        for i in range(1, len(self.bi_points)):
            a, b = self.bi_points[i - 1], self.bi_points[i]
            if a["type"] == "top" and b["type"] == "bottom":
                if a["price"] <= b["price"]:
                    violations.append({
                        "contract": contract,
                        "rule": "bi_direction",
                        "detail": f"down-bi [{i-1}→{i}] top={a['price']} <= bottom={b['price']}",
                        "severity": "MEDIUM"
                    })
            elif a["type"] == "bottom" and b["type"] == "top":
                if b["price"] <= a["price"]:
                    violations.append({
                        "contract": contract,
                        "rule": "bi_direction",
                        "detail": f"up-bi [{i-1}→{i}] top={b['price']} <= bottom={a['price']}",
                        "severity": "MEDIUM"
                    })

        # 4. Pivot zg > zd
        for i, p in enumerate(self.pivots):
            if p["zg"] <= p["zd"]:
                violations.append({
                    "contract": contract,
                    "rule": "pivot_zg_gt_zd",
                    "detail": f"pivot[{i}] zg={p['zg']} <= zd={p['zd']}",
                    "severity": "HIGH"
                })

        # 5. Inclusion residual: check if consecutive k_lines still have inclusion
        inclusion_violations = 0
        for i in range(1, len(self.k_lines)):
            a, b = self.k_lines[i - 1], self.k_lines[i]
            if (b["high"] <= a["high"] and b["low"] >= a["low"]) or \
               (a["high"] <= b["high"] and a["low"] >= b["low"]):
                inclusion_violations += 1
        if inclusion_violations > 0:
            violations.append({
                "contract": contract,
                "rule": "inclusion_residual",
                "detail": f"{inclusion_violations} pairs still have inclusion after processing",
                "severity": "MEDIUM"
            })

        # 6. Pivot overlap check: consecutive pivots with overlapping ranges
        pivot_overlaps = 0
        for i in range(1, len(self.pivots)):
            p1, p2 = self.pivots[i - 1], self.pivots[i]
            overlap = min(p1["zg"], p2["zg"]) > max(p1["zd"], p2["zd"])
            if overlap:
                pivot_overlaps += 1
        if pivot_overlaps > 0:
            violations.append({
                "contract": contract,
                "rule": "pivot_overlap",
                "detail": f"{pivot_overlaps}/{len(self.pivots)-1} consecutive pivot pairs overlap (may indicate missed merge/upgrade)",
                "severity": "LOW"
            })

        return violations


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    all_violations = []
    summary = []

    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            print(f"WARNING: no file for {c}")
            continue
        fp = matches[0]

        df = pd.read_csv(fp)
        df.columns = [col.strip() for col in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        ext = ChanStructureExtractor(df)
        vs = ext.check_invariants(c.upper())
        all_violations.extend(vs)

        summary.append({
            "contract": c.upper(),
            "raw_5m_bars": len(ext.df_5m),
            "k_lines_after_inclusion": len(ext.k_lines),
            "bi_points": len(ext.bi_points),
            "pivots": len(ext.pivots),
            "violations": len(vs),
        })

        print(f"\n{'='*60}")
        print(f"{c.upper()} | 5m bars: {len(ext.df_5m)} | k_lines: {len(ext.k_lines)} | bi: {len(ext.bi_points)} | pivots: {len(ext.pivots)}")
        if vs:
            for v in vs:
                print(f"  [{v['severity']}] {v['rule']}: {v['detail']}")
        else:
            print("  All invariants passed ✓")

    # Summary
    print(f"\n\n{'='*70}")
    print("INVARIANT CHECK SUMMARY")
    print(f"{'='*70}")
    for s in summary:
        flag = "OK" if s["violations"] == 0 else f"FAIL ({s['violations']})"
        print(f"  {s['contract']}: {flag} | 5m={s['raw_5m_bars']} -> k={s['k_lines_after_inclusion']} -> bi={s['bi_points']} -> pivots={s['pivots']}")

    # Violation breakdown by rule
    if all_violations:
        print(f"\nViolation breakdown:")
        from collections import Counter
        by_rule = Counter(v["rule"] for v in all_violations)
        for rule, count in by_rule.most_common():
            print(f"  {rule}: {count}")


if __name__ == "__main__":
    main()
