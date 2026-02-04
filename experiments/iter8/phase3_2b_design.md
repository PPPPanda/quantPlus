# Iter8 Phase3 — 2B/2S 改造：从“趋势跟随加仓”到“走势段背驰框架”

> 目标：把当前 2B/2S 的触发从「低点抬高 + diff 增强 + 中枢存在」的趋势跟随式加仓，改为缠论更标准的 **走势段（segment）力度比较**：
> - **2B（买点）**：下跌段（离开中枢的一段）再创新低/或形成底分型时，**力度衰减**（背驰）→ 反向入场（做多）
> - **2S（卖点）**：上涨段（离开中枢的一段）再创新高/或形成顶分型时，**力度衰减**（背驰）→ 平多
>
> 约束对齐：只做多；不全局降频（只优化信号判定）；不收紧 trailing（activate>=2.0 保持）；确保 p2601 Sharpe>=2.5，p2209不大幅退化。

---

## 0. 总体思路（为什么现有 2B 会过度交易）

现有逻辑本质是：
- 在存在 forming/active 中枢时，看到「新低点抬高」且 diff 变强，就直接 Buy。

问题：
- 在 **震荡/箱体** 中，低点抬高与 diff 变强会在多个小波动里反复出现；
- 这更像“顺势确认 + 趋势跟随加仓”，不是背驰；
- 缠论背驰强调的是：**同级别走势段力度对比**（通常比较 MACD 面积/能量），出现“新低/新高，但能量变弱”的衰竭信号。

因此改造关键：
1) 把比较对象从“相邻两个笔端点”升级为“相邻两段同向走势段（segment）”；
2) 力度指标从瞬时 diff 变成：**该段内部（多笔）MACD histogram 面积总和**（能量）。

---

## 1) 走势段（segment）的定义：如何从 `bi_points` 提取同向走势段

### 1.1 输入数据回顾
- `self._bi_points`: 笔端点序列，点结构：`{type: 'top'|'bottom', price, idx, data}`。
  - 点序列理论上 top/bottom 交替。
- `self._bi_macd_areas`: 每笔完成时的 MACD histogram 面积（建议保存为 **绝对面积** 或按方向处理，见 2.2）。
  - 一般长度≈笔数≈`len(bi_points)-1`。

### 1.2 段的抽取目标
我们需要一个“同向段”的抽取规则，满足：
- 一段由**连续多笔**组成；
- 段的方向一致：
  - **down 段**：从某个 top 开始，经历 bottom/top/bottom...，整体低点不断下移（或至少创新低），直到出现明显反向突破/回抽结构；
  - **up 段**：从某个 bottom 开始，整体高点不断上移。

### 1.3 可落地的简化段定义（适配现有数据结构）
在不引入“线段分解算法/特征序列严格判定”的前提下，可用 **三点法 + 方向一致性** 抽段：

**定义 A（工程可实现、足够抑制震荡重复触发）**
- 取 `bi_points` 中最近的端点，构造“段候选”时只在端点处更新。
- 段以“关键极值被打破”为结束条件：
  - down 段：起点为最近一个 `top`，结束条件是后续出现 `top` 且其 price **突破**前一段下跌过程中的关键回抽高点（或更简单：突破 down 段起点 top 的某个比例回撤阈值）。
  - up 段：起点为最近一个 `bottom`，结束条件是后续出现 `bottom` 且其 price **跌破**前一段上涨过程中的关键回抽低点。

**实现上的最简段抽取（推荐）：用“突破上一个反向端点”判定段切换**
- 笔端点交替，段方向可用最近 3 个端点确定：
  - 若 `p[-3]` 为 top，`p[-2]` 为 bottom，`p[-1]` 为 top：
    - 若 `p[-1].price < p[-3].price` → 仍偏弱（down 段中的回抽 top 走低）
    - 若 `p[-1].price > p[-3].price` → 回抽 top 走高（有段切换可能）
  - 对 bottom 同理。

为了稳健与可控，建议构造段时使用以下 **段切换判定**：
- **down 段持续条件**：回抽 top 不创新高（top 逐步走低/不突破上一个 top）
- **down→up 切换**：出现一个回抽 top `T_new`，满足 `T_new.price > T_prev.price` 且同时其后出现 bottom 不再创新低（或直接：`T_new` 突破了 down 段内的“最后一个 top”）。

> 备注：这不是严格缠论线段，但对“做走势段背驰”足够：因为背驰比较只需两段同向（两段下跌或两段上涨）能量对比。

### 1.4 段的数据结构建议
输出段列表 `segments`，每段：
```python
{
  'dir': 'down'|'up',
  'start_point_i': int,   # 在 bi_points 中的起点索引
  'end_point_i': int,     # 在 bi_points 中的终点索引（段结束端点）
  'start_idx': int,       # K线 idx
  'end_idx': int,
  'start_price': float,
  'end_price': float,
  'bi_start': int,        # 笔索引起点（对应 bi_macd_areas 的下标）
  'bi_end': int,          # 笔索引终点（包含/不包含需统一）
  'macd_area': float,     # 段内 MACD histogram 面积（见第2节）
  'bars': int,            # 可选：段跨越bar数
  'bi_count': int,
}
```

---

## 2) 力度比较方法：MACD 面积比较的具体实现

### 2.1 为什么用“段内面积”而不是 diff 强弱
- diff 在端点处是“瞬时状态”，对震荡非常敏感；
- 缠论的力度/能量更接近“这一段走出来用了多少动能”，面积比单点更稳定；
- 现有 `self._bi_macd_areas` 已经是“每笔完成时的 histogram 面积”，天然适合段内求和。

### 2.2 面积的方向与归一化（关键实现细节）
你现在的 `self._bi_macd_areas` 未说明是否含符号。建议采取以下口径之一（推荐 2.2.2）：

#### 2.2.1 若 `bi_macd_areas` 已按方向带符号
- up 笔面积为正，down 笔面积为负。
- 则段内力度可用：
  - `seg_area = abs(sum(areas[bi_start:bi_end]))`

#### 2.2.2 若 `bi_macd_areas` 是绝对值（或符号不可靠）【推荐】
- 统一以段方向过滤：
  - up 段：只统计 histogram>0 的面积（或只统计“上行笔”面积）
  - down 段：只统计 histogram<0 的面积的绝对值（或只统计“下行笔”面积）

考虑你已有 `diff_5m, dea_5m`，hist = diff-dea；若每根bar可得 hist，可更精细。但目前只到“每笔面积”，则用“笔方向”更稳：
- 笔方向可由相邻端点 price 变化确定：
  - 从 bottom→top 为 up 笔
  - 从 top→bottom 为 down 笔

**段内力度（推荐实现）**
- `seg_area = sum(abs(area_k) for k in bi_range if bi_dir(k)==seg_dir)`

### 2.3 背驰（力度衰减）的量化判定
我们需要比较两个同向段：
- 对 2B：比较最近两段 **down 段**：`D1`（前一段下跌）与 `D2`（当前下跌）
  - 价格创新低（或至少接近新低），但 `area(D2) < area(D1)` → 下跌力度衰减。
- 对 2S：比较最近两段 **up 段**：`U1` 与 `U2`
  - 价格创新高，但 `area(U2) < area(U1)` → 上涨力度衰减。

建议使用 **双阈值**：相对衰减 + 最小能量过滤，避免小噪声：
- 相对衰减：
  - `area_ratio = area2 / (area1 + eps)`
  - 背驰成立：`area_ratio <= r_th`，建议 `r_th` 初值 0.75~0.85
- 最小能量过滤：
  - `area1 >= area_min`（上一段要“真用力”）
  - `area2 >= area2_min` 或允许很小（衰竭），但需配合结构确认（见 3.2）

同时加入 **价格条件**：
- 2B：`low2 <= low1 * (1 + tol)` 或 “创新低/等低”，tol=0.1%~0.3%（合约不同可自适应到 ATR）。
- 2S：`high2 >= high1 * (1 - tol)`。

---

## 3) 2B 信号触发条件：完整条件链（只做多）

> 目标：把 2B 变成「离开中枢后的下跌段背驰 + 底分型确认 + 重新回到中枢/站回关键位」的组合；既不全局降频，又能在震荡减少重复。

### 3.1 结构前提（中枢/走势环境）
保持你现有的“有结构”前提，但更精准：
- `ap = self._active_pivot` 存在，且 `ap['state'] in ('forming', 'active')`
- 当前价格相对于中枢：
  - **优先触发场景**：从中枢向下离开（price < zd 一段），再出现背驰。

实现可用：
- `is_below_pivot = last_price < ap['zd']`
- 或者以端点判断：最近 bottom 端点 price < ap['zd']。

### 3.2 背驰核心：同向段力度衰减
当最新端点 `p_now` 为 bottom（形成底分型/下跌端点完成）时，计算最近两个 down 段 `D1, D2`：
- `D2.dir == 'down'` 且 `D1.dir == 'down'`
- 价格条件：`D2.end_price <= D1.end_price * (1 + tol)`（创新低/等低）
- 力度条件：`D2.macd_area / (D1.macd_area + eps) <= r_th`
- 结构过滤：`D1.macd_area >= area_min`（防止两段都很小的震荡）

### 3.3 确认条件（避免“下跌中继”误判）
仅靠面积衰减仍可能在趋势中继产生早抄底。加入“确认”但不做全局降频：

**确认 A（底分型已完成）**
- 你已有 `bi_points`，当 `p_now.type == 'bottom'` 时天然是笔端点确认。

**确认 B（回拉）**：背驰后至少出现一笔上行确认
- 触发时机可有两种：
  1) **立即触发**：在 bottom 端点处触发，但要求当前 bar 的 hist 已向上收敛：`(diff_5m - dea_5m) > (prev_hist)` 或 hist 绝对值收敛；
  2) **一笔确认触发（更稳）**：在随后出现一个 top 端点（即 bottom→top 的上行笔完成）且该 top 回到中枢下沿附近时触发。

考虑“不全局降频”，推荐采用折中：
- 允许 bottom 处触发，但必须满足：
  - `hist_now > hist_min`（例如 hist 从负值快速抬升，或 hist 由负转正的早期）
  - 或 `last_price > D2.end_price + k*ATR`（小幅反弹确认）

**确认 C（回到中枢）**：尽量要求“向中枢回拉”
- `last_price >= ap['zd']` 或最近上行笔高点触及 `ap['zd']`。
- 这样能显著减少震荡里“中枢下沿附近来回戳”的重复进出：
  - 真背驰往往伴随回拉重返中枢。

### 3.4 最终 2B 条件链（建议）
**2B = 结构 + 离开中枢 + 下跌段背驰 + 确认**
1. `ap.state in ('forming','active')`
2. 最近完成端点 `p_now.type == 'bottom'`
3. `is_bull`（你原逻辑中有趋势过滤则保留，但建议改为更慢的多头环境过滤，避免震荡里频繁开仓；例如更高周期均线向上）
4. `below_pivot`：`p_now.price < ap.zd`（离开中枢向下）
5. `down_divergence(D1,D2)`：价格创新低/等低 且 面积衰减
6. `confirm`：满足任一：
   - `hist` 收敛上拐（diff-dea 上升、或负值绝对值下降）
   - 或价格回到 `ap.zd` 上方（回中枢）

触发：`sig='Buy'`。

---

## 4) 2S 信号触发条件（只平多，不做空）

2S 用对称逻辑：**上涨段背驰** → 平多。

### 4.1 结构前提
- 有持仓（long position > 0）
- 最好仍要求中枢结构存在：
  - `ap.state in ('forming','active')`（或最近归档中枢也可）

### 4.2 背驰核心
当 `p_now.type == 'top'`：取最近两段 up 段 `U1, U2`
- 价格条件：`U2.end_price >= U1.end_price*(1 - tol)`（创新高/等高）
- 力度条件：`U2.area / (U1.area+eps) <= r_th_sell`（可略严一点，比如 0.70~0.80）
- 能量过滤：`U1.area >= area_min`

### 4.3 确认条件
- 2S 可以更快（因为只是平多，风险更小），但避免“强趋势里过早离场”。
- 建议确认：
  - 顶分型完成（p_now==top）
  - 且 hist 开始回落（hist_now < hist_prev 或由正收敛）
  - 或价格跌回中枢上沿 `zg` 下方（回中枢）

触发：`sig='Sell'`（只做平仓）。

---

## 5) 伪代码（可直接翻译为 Python）

### 5.1 段构建：从 bi_points + bi_macd_areas 生成 segments

> 注意：以下为“工程化段”，目标是稳定可用、能做两段同向比较。若后续要更严格缠论线段，可替换 segment_builder，而不影响 2B/2S 主体。

```python
def _build_segments(self, bi_points, bi_macd_areas):
    # bi_points: [P0, P1, ...] alternating top/bottom
    # bi_macd_areas: areas for each bi (P0->P1 is bi0, P1->P2 is bi1, ...)

    segments = []
    if len(bi_points) < 5:
        return segments

    # helper: direction of a bi by its endpoints
    def bi_dir(i):
        # bi i connects points i -> i+1
        return 'up' if bi_points[i+1]['price'] > bi_points[i]['price'] else 'down'

    # helper: accumulate macd area for a segment over bi indices
    def seg_area(seg_dir, bi_i0, bi_i1_excl):
        s = 0.0
        for k in range(bi_i0, bi_i1_excl):
            if bi_dir(k) != seg_dir:
                continue
            a = bi_macd_areas[k]
            s += abs(a)
        return s

    # ---- segment finding (simple swing-based) ----
    # We define segment turns using break of prior same-type extreme.

    # start from first bi
    cur = {
        'dir': bi_dir(0),
        'start_point_i': 0,
        'bi_start': 0,
    }

    # maintain last same-type extreme for turn detection
    # for cur.dir='down', watch tops: if a new top breaks above previous top => possible turn
    last_top_price = None
    last_bottom_price = None

    for i in range(1, len(bi_points)):
        p = bi_points[i]
        if p['type'] == 'top':
            if last_top_price is None:
                last_top_price = p['price']
            else:
                # if current segment is down, and we see higher high at top => turn up
                if cur['dir'] == 'down' and p['price'] > last_top_price:
                    # close segment at previous point (i-1)
                    end_point_i = i-1
                    bi_end = end_point_i  # bi index exclusive == end_point_i
                    segments.append({
                        'dir': cur['dir'],
                        'start_point_i': cur['start_point_i'],
                        'end_point_i': end_point_i,
                        'start_idx': bi_points[cur['start_point_i']]['idx'],
                        'end_idx': bi_points[end_point_i]['idx'],
                        'start_price': bi_points[cur['start_point_i']]['price'],
                        'end_price': bi_points[end_point_i]['price'],
                        'bi_start': cur['bi_start'],
                        'bi_end': bi_end,
                        'macd_area': seg_area(cur['dir'], cur['bi_start'], bi_end),
                        'bi_count': bi_end - cur['bi_start'],
                    })
                    # start new up segment
                    cur = {'dir': 'up', 'start_point_i': end_point_i, 'bi_start': bi_end}
                last_top_price = p['price']

        elif p['type'] == 'bottom':
            if last_bottom_price is None:
                last_bottom_price = p['price']
            else:
                # if current segment is up, and we see lower low at bottom => turn down
                if cur['dir'] == 'up' and p['price'] < last_bottom_price:
                    end_point_i = i-1
                    bi_end = end_point_i
                    segments.append({
                        'dir': cur['dir'],
                        'start_point_i': cur['start_point_i'],
                        'end_point_i': end_point_i,
                        'start_idx': bi_points[cur['start_point_i']]['idx'],
                        'end_idx': bi_points[end_point_i]['idx'],
                        'start_price': bi_points[cur['start_point_i']]['price'],
                        'end_price': bi_points[end_point_i]['price'],
                        'bi_start': cur['bi_start'],
                        'bi_end': bi_end,
                        'macd_area': seg_area(cur['dir'], cur['bi_start'], bi_end),
                        'bi_count': bi_end - cur['bi_start'],
                    })
                    cur = {'dir': 'down', 'start_point_i': end_point_i, 'bi_start': bi_end}
                last_bottom_price = p['price']

    # close last open segment at latest confirmed point
    end_point_i = len(bi_points) - 1
    bi_end = end_point_i
    if bi_end > cur['bi_start']:
        segments.append({
            'dir': cur['dir'],
            'start_point_i': cur['start_point_i'],
            'end_point_i': end_point_i,
            'start_idx': bi_points[cur['start_point_i']]['idx'],
            'end_idx': bi_points[end_point_i]['idx'],
            'start_price': bi_points[cur['start_point_i']]['price'],
            'end_price': bi_points[end_point_i]['price'],
            'bi_start': cur['bi_start'],
            'bi_end': bi_end,
            'macd_area': seg_area(cur['dir'], cur['bi_start'], bi_end),
            'bi_count': bi_end - cur['bi_start'],
        })

    return segments
```

> 说明：
> - 这套段切换逻辑偏保守，能显著减少震荡里段的频繁切换，从而减少背驰触发频率。
> - 如果发现趋势行情中反应慢，可把“突破判定”改为更敏捷的（例如仅需突破前一个同型端点而非全局 last_top/last_bottom），或加回撤阈值。

### 5.2 背驰判定函数
```python
def _is_divergence(seg1, seg2, tol=0.002, ratio_th=0.8, eps=1e-9, area_min=1.0):
    # seg1: older, seg2: newer, same direction
    if seg1 is None or seg2 is None:
        return False
    if seg1['dir'] != seg2['dir']:
        return False
    if seg1['macd_area'] < area_min:
        return False

    # price condition depends on direction
    if seg2['dir'] == 'down':
        price_ok = seg2['end_price'] <= seg1['end_price'] * (1 + tol)  # new low / equal low
    else:
        price_ok = seg2['end_price'] >= seg1['end_price'] * (1 - tol)  # new high / equal high

    if not price_ok:
        return False

    ratio = seg2['macd_area'] / (seg1['macd_area'] + eps)
    return ratio <= ratio_th
```

### 5.3 2B/2S 主判定伪代码（替换当前 2B/2S 逻辑）

```python
def _check_signal(self):
    sig = None

    ap = self._active_pivot
    has_active_structure = (ap is not None and ap.get('state') in ('forming', 'active'))

    bi_points = self._bi_points
    if len(bi_points) < 6:
        return None

    p_now = bi_points[-1]

    # build segments lazily / cached
    segments = self._build_segments(bi_points, self._bi_macd_areas)
    if len(segments) < 4:
        return None

    # helper: get last two segments of a direction
    def last_two(dir_):
        xs = [s for s in segments if s['dir'] == dir_]
        if len(xs) < 2:
            return None, None
        return xs[-2], xs[-1]

    hist_now = self.diff_5m[-1] - self.dea_5m[-1]
    hist_prev = self.diff_5m[-2] - self.dea_5m[-2]

    # --- 2B: Buy (only long entry) ---
    if sig is None and has_active_structure:
        if p_now['type'] == 'bottom':
            # environment: pivot leave-down
            below_pivot = (p_now['price'] < ap['zd'])

            D1, D2 = last_two('down')
            div_ok = _is_divergence(D1, D2, tol=0.002, ratio_th=0.82, area_min=self._div_area_min)

            # confirmation (choose one of two gates)
            hist_rebound = (hist_now > hist_prev)  # histogram contracting / turning up
            back_to_pivot = (self.close[-1] >= ap['zd'])

            if below_pivot and div_ok and (hist_rebound or back_to_pivot) and self.is_bull:
                sig = 'Buy'

    # --- 2S: Sell (only exit long) ---
    if sig is None and self.pos > 0 and has_active_structure:
        if p_now['type'] == 'top':
            U1, U2 = last_two('up')
            div_ok = _is_divergence(U1, U2, tol=0.002, ratio_th=0.78, area_min=self._div_area_min)

            hist_fall = (hist_now < hist_prev)
            back_to_pivot = (self.close[-1] <= ap['zg'])

            if div_ok and (hist_fall or back_to_pivot):
                sig = 'Sell'

    return sig
```

参数建议（首轮回测建议从宽到严微调）：
- `ratio_th`：2B 0.80~0.85；2S 0.75~0.80
- `tol`：0.1%~0.3% 或用 ATR 自适应（不同合约波动不同）
- `area_min`：用分位数自适应（例如过去 N 段 area 的 30% 分位）比固定值更稳

---

## 6) 风险分析：可能影响哪些合约、如何防护

### 6.1 预期收益/交易次数变化
- **震荡合约（容易过度交易）**：会显著减少 2B 触发次数，降低手续费与反复止损。
  - 你的现状里 p2501 81 笔偏多，这个框架大概率能把频率压下来，同时维持 PF。
- **趋势强、回撤少的合约**：背驰触发会变少，可能错过部分顺势加仓收益。
  - 但你要求不收紧 trailing（activate>=2.0），且 2B 原本在趋势里更像加仓；改造后可能减少加仓次数。
  - 若回测显示趋势合约收益下降，可在 2B 之外保留单独的“趋势加仓”模块，但必须与背驰信号隔离（避免又回到原问题）。

### 6.2 p2209 不退化：主要风险点
p2209 当前 PF≈1.06、略靠“多触发 + 运气”撑住。
- 改为段背驰后，信号数量下降，可能导致：
  1) 机会减少 → 总收益下降；
  2) 如果过滤太严，可能只剩少量信号且质量不够 → PF 下降。

**防护措施：**
- 对 p2209 采用更宽松的背驰阈值（ratio_th 更高，如 0.85），或降低 `area_min`；
- 允许“等低背驰”（不要求严格创新低，只要求接近前低 `tol` 稍放大）；
- 对 2B 的确认条件选择 `hist_rebound OR back_to_pivot`（而不是 AND），保证不降频过猛。

### 6.3 p2301 当前亏损：可能改善但也有结构风险
p2301 PF 0.75，可能震荡噪声更强、趋势更弱。
- 段背驰 + 回中枢确认，有望减少错误开仓；
- 但若该合约走势经常“弱反弹后继续下跌”，背驰抄底仍会中枪。

**防护措施：**
- 对 2B 增加一个“最小反弹确认”：例如背驰后至少出现一笔 up 笔完成（bottom→top）再入场（会略降频，但对亏损合约有必要）；
- 或要求回到 `ap.zd` 再入场（更强过滤）。

### 6.4 工程实现风险（数据结构一致性）
- `bi_macd_areas` 与 `bi_points` 对齐必须严格：
  - 若 `areas[i]` 对应 `bi_points[i] -> bi_points[i+1]`，则 `bi_end` 的含义要统一。
- 段构建算法如果过于敏感，会导致段过短 → 背驰变成“笔级别比较”，又回到噪声。

**防护措施：**
- 段的最小笔数：`seg.bi_count >= 3` 才纳入背驰比较；
- 段能量的最小阈值：`area_min` 用历史分位数自适应；
- 回测时重点观察：
  - 每合约 2B 次数是否显著下降（尤其 p2501）
  - p2601 Sharpe 是否保持≥2.5
  - p2209 的 trade count 是否下降过多

---

## 7) 与现有逻辑的替换点（给实现者的提示）

你当前 2B：
- 只用 `p_prev/p_now` + diff 强弱。

替换方案：
1) 在每次 `_check_signal()` 或每笔完成时构建/更新 `segments`（建议缓存，端点新增时增量更新）；
2) 2B/2S 触发都以 **最近端点 + 最近两段同向段** 来判定；
3) 仍可保留 `has_active_structure` 作为结构门槛，但把核心从 diff 单点比较改为段面积比较。

---

## 8) 回测与调参建议（简要）
- 第一轮：
  - ratio_th_buy=0.82, ratio_th_sell=0.78, tol=0.002
  - confirm：`hist_rebound OR back_to_pivot`
  - seg_min_bi=3
- 若交易仍偏多（震荡触发）：
  - 提高 seg_min_bi 或提高 area_min（用分位数）
  - 强化 back_to_pivot 条件（buy 必须回到 zd）
- 若 p2209 交易太少：
  - 放宽 ratio_th_buy 到 0.85；tol 放到 0.003；area_min 下调

