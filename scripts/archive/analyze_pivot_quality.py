"""analyze_pivot_quality.py

Deep analysis of pivot (zhongshu) quality issues:
1. How many pivots overlap with the next one? (should be merged/extended)
2. What's the typical pivot lifespan (bi count from creation to replacement)?
3. How does "last pivot only" affect signal quality?
4. Simulate: if we merge overlapping pivots, how many real pivots remain?
"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class PivotAnalyzer:
    def __init__(self, df_1m: pd.DataFrame, min_bi_gap: int = 4):
        self.min_bi_gap = min_bi_gap
        df_idx = df_1m.set_index("datetime")
        df_idx.index = pd.to_datetime(df_idx.index)
        self.df_5m = (
            df_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots_raw = []  # All pivots (sliding window)
        self._build()

    def _build(self):
        for ts, row in self.df_5m.iterrows():
            bar = {"high": float(row["high"]), "low": float(row["low"]), "time": ts}
            self._inclusion(bar)
            self._bi()

        # Build ALL pivots from bi_points
        for i in range(3, len(self.bi_points)):
            pts = self.bi_points[i-3:i+1]
            ranges = [(min(pts[j]["price"], pts[j+1]["price"]),
                        max(pts[j]["price"], pts[j+1]["price"])) for j in range(3)]
            zg = min(r[1] for r in ranges)
            zd = max(r[0] for r in ranges)
            if zg > zd:
                self.pivots_raw.append({
                    "zg": zg, "zd": zd,
                    "start_bi": i-3, "end_bi": i,
                    "bi_idx_range": (i-3, i)
                })

    def _inclusion(self, nb):
        if not self.k_lines:
            self.k_lines.append(nb); return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
            m = last.copy(); m["time"] = nb["time"]
            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"]); m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"]); m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]: self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]: self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self):
        if len(self.k_lines) < 3: return
        c, m2, l = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(self.k_lines)-2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(self.k_lines)-2}
        if not cand: return
        if not self.bi_points:
            self.bi_points.append(cand); return
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= self.min_bi_gap:
                self.bi_points.append(cand)

    def merge_overlapping_pivots(self) -> List[Dict]:
        """Merge consecutive pivots that overlap into extended pivots."""
        if not self.pivots_raw:
            return []
        merged = [self.pivots_raw[0].copy()]
        for p in self.pivots_raw[1:]:
            last = merged[-1]
            # Check overlap
            overlap_zg = min(last["zg"], p["zg"])
            overlap_zd = max(last["zd"], p["zd"])
            if overlap_zg > overlap_zd:
                # Merge: extend the pivot
                last["zg"] = overlap_zg  # narrower range (intersection)
                last["zd"] = overlap_zd
                last["end_bi"] = p["end_bi"]
            else:
                merged.append(p.copy())
        return merged

    def analyze(self, contract: str):
        raw_count = len(self.pivots_raw)
        merged = self.merge_overlapping_pivots()
        merged_count = len(merged)

        # Pivot lifespan: how many bi points from creation to being replaced
        lifespans = []
        for i in range(1, len(self.pivots_raw)):
            life = self.pivots_raw[i]["end_bi"] - self.pivots_raw[i-1]["end_bi"]
            lifespans.append(life)

        # Overlap stats
        overlaps = 0
        for i in range(1, len(self.pivots_raw)):
            a, b = self.pivots_raw[i-1], self.pivots_raw[i]
            if min(a["zg"], b["zg"]) > max(a["zd"], b["zd"]):
                overlaps += 1

        # Merged pivot sizes (how many raw pivots each merged one contains)
        merged_sizes = []
        if merged:
            j = 0
            for m in merged:
                count = 0
                while j < len(self.pivots_raw) and self.pivots_raw[j]["end_bi"] <= m["end_bi"]:
                    count += 1
                    j += 1
                merged_sizes.append(count)

        print(f"\n{'='*60}")
        print(f"{contract}")
        print(f"  Bi points: {len(self.bi_points)}")
        print(f"  Raw pivots (sliding window): {raw_count}")
        print(f"  Overlapping pairs: {overlaps}/{max(raw_count-1,1)} ({overlaps/max(raw_count-1,1)*100:.0f}%)")
        print(f"  Merged pivots: {merged_count} (reduction: {raw_count-merged_count}, {(raw_count-merged_count)/max(raw_count,1)*100:.0f}%)")
        if lifespans:
            print(f"  Pivot lifespan (bi): median={np.median(lifespans):.0f}, mean={np.mean(lifespans):.1f}, max={max(lifespans)}")
        if merged_sizes:
            print(f"  Merged pivot size: median={np.median(merged_sizes):.0f}, mean={np.mean(merged_sizes):.1f}, max={max(merged_sizes)}")
            # How many merged pivots are "big" (>= 5 raw pivots)
            big = sum(1 for s in merged_sizes if s >= 5)
            print(f"  Big merged pivots (>=5 raw): {big}/{merged_count}")

        return {
            "contract": contract,
            "bi": len(self.bi_points),
            "raw_pivots": raw_count,
            "merged_pivots": merged_count,
            "overlap_pct": overlaps/max(raw_count-1,1)*100,
            "median_lifespan": np.median(lifespans) if lifespans else 0,
        }


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    results = []
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches: continue
        df = pd.read_csv(matches[0])
        df.columns = [col.strip() for col in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        analyzer = PivotAnalyzer(df)
        r = analyzer.analyze(c.upper())
        results.append(r)

    print(f"\n\n{'='*70}")
    print("PIVOT QUALITY SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['contract']}: raw={r['raw_pivots']} -> merged={r['merged_pivots']} | overlap={r['overlap_pct']:.0f}% | lifespan={r['median_lifespan']:.0f} bi")


if __name__ == "__main__":
    main()
