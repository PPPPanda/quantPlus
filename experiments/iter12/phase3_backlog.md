# Iteration 12 Phase 3 — Backlog（Claude主控初版）

基于 Phase 1 三方审计 + Phase 2 失败模式诊断

---

## 低改动半径（Phase 4 优先执行）

### B1: 止损buffer提升 [FM-4]
- **失败模式**: 止损过紧，低波动合约频繁被扫
- **目标**: p2401, p2201 止损不被无谓扫掉
- **方案**: `stop_buffer_atr_pct` 从 0.02 → 0.05（或0.04/0.06网格搜索）
- **缠论依据**: Phase 1审计项7——止损应在结构破坏点，不是噪音区
- **代码改动**: 仅改参数，无逻辑变化
- **预期影响**: 
  - p2401/p2201: 减少短止损被扫，改善单笔盈亏比
  - 风险：止损加宽可能增大单笔亏损尾部
- **回归检查**: p2209/p2601 不退化

### B2: ATR波动率门槛 [FM-2]
- **失败模式**: 低波动过度交易（p2401）
- **目标**: 在ATR过低时不开仓
- **方案**: 新增参数 `min_atr_pct`，计算 `atr / close`，当该比例低于门槛时跳过Buy信号
- **缠论依据**: 第25课——区间套，当走势区间过小时不应参与（力度不足）
- **代码改动**: 在 `_check_signal()` 中 Buy 信号生成前加：
  ```python
  if self.min_atr_pct > 0 and self.atr > 0:
      atr_pct = self.atr / curr_bar['close']
      if atr_pct < self.min_atr_pct:
          return  # 波动率过低，跳过
  ```
- **参数候选**: min_atr_pct=0.005（ATR占价格0.5%）起步，网格搜索0.003-0.008
- **预期影响**: 
  - p2401: 大幅减少无效交易
  - 风险：可能过滤掉低波动但有效的入场（如p2309震荡市）
- **回归检查**: p2309（震荡市正收益合约）不退化

### B3: 递进式断路器 [FM-3]
- **失败模式**: 超长连亏（p2305的12笔连亏突破cb=6限制）
- **目标**: 断路器暂停后如果恢复又亏，加大暂停时长
- **方案**: 新增参数 `cb_escalation_mult`（默认1.5），每次触发断路器后暂停时间递增：
  - 第1次: 60 bars
  - 第2次: 90 bars (60 * 1.5)
  - 第3次: 135 bars (90 * 1.5)
- **代码改动**: `_update_loss_streak()` 中增加计数器 `_cb_trigger_count`
  ```python
  if self._consecutive_losses >= self.circuit_breaker_losses:
      self._cb_trigger_count += 1
      pause = int(self.circuit_breaker_bars * (self.cb_escalation_mult ** (self._cb_trigger_count - 1)))
      self._cooldown_remaining = pause
  ```
  盈利时重置 `_cb_trigger_count = 0`
- **预期影响**: 
  - p2305: 更有效阻断超长连亏
  - 风险：在趋势合约上暂停太久可能错过趋势
- **回归检查**: p2209/p2601 的趋势行情不能因暂停太久被错过

### B4: 预热期 [FM-1]
- **失败模式**: 合约初期指标未稳定时的连亏
- **目标**: 开盘前N个5m bar不开仓
- **方案**: 新增参数 `warmup_bars`（默认60 = 5小时），在 `_bar_5m_count < warmup_bars` 时不生成信号
- **代码改动**: `_check_signal()` 头部加：
  ```python
  if self.warmup_bars > 0 and self._bar_5m_count < self.warmup_bars:
      return
  ```
- **预期影响**: 
  - p2201: 跳过2021-08的初始亏损期
  - p2309: 跳过2023-04的初始亏损期
  - 风险：如果趋势在初期就启动，可能错过
- **回归检查**: p2209（2022-04就开始趋势）不能因预热错过主升浪

---

## 中改动半径（Phase 4 次优先）

### B5: S6面积背驰强化 [Phase1-审计项5/6]
- **失败模式**: S6路径被限制太紧（div_threshold=0.50），且缠论依据更充分
- **目标**: 增加有效的底背驰入场机会
- **方案**: 降低 `div_threshold` 从 0.50 → 0.35-0.40
- **代码改动**: 仅改参数
- **预期影响**: 
  - 更多S6信号 → 低位入场 → 改善盈亏比
  - 风险：门槛太低导致虚假背驰信号增多

### B6: 15m MACD 0轴过滤 [Phase1-审计项6]
- **失败模式**: 15m过滤过于简化，is_bull在低位震荡时频繁切换
- **目标**: 只在15m MACD处于健康多头（DIF>0轴附近）时才做多
- **方案**: 修改 `is_bull` 条件，加入力度门槛
  ```python
  # 方案A: DIF > DEA 且 DIF > 0
  is_bull = self._prev_diff_15m > self._prev_dea_15m and self._prev_diff_15m > 0
  
  # 方案B: 柱状图 > 0（更宽松）
  is_bull = (self._prev_diff_15m - self._prev_dea_15m) > 0
  ```
- **预期影响**:
  - p2401: 减少15m弱势时的虚假多头信号
  - 风险：过滤太严会杀趋势初期的入场（p2209可能受伤）

---

## 执行顺序

Phase 4 小循环安排：
1. **Round 1**: B1（止损buffer=0.05）→ 13合约回测
2. **Round 2**: B2（ATR波动率门槛 min_atr_pct=0.005）→ 13合约回测
3. **Round 3**: B3（递进式断路器 cb_escalation_mult=1.5）→ 13合约回测
4. **Round 4**: B4（预热期 warmup_bars=60）→ 13合约回测
5. **Round 5**: B1+B2 最优组合 → 13合约回测

每轮：
- 只改1个变量
- 记录 config + metrics + changelog
- 对比 baseline_13.json
- 检查门槛：TOTAL >= 12164, 所有负合约 > -180

---

*等待 GPT-5.2 Phase 3 方案后合并更新*
