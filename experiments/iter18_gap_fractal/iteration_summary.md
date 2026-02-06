# iter18: 节假日跳空对缠论分型影响研究

## 状态
- Phase 1 ✅ 完成（结构审计）
- Phase 2 ✅ 完成（数据分析）
- Phase 3 ✅ 完成（改动方案）
- Phase 4 🔄 代码已写，回测待运行（venv 锁定问题）
- Phase 5 ⏳ 待 Phase 4 完成后审核

## 关键发现

### 数据分析（119个节假日跳空，>3 ATR，>48h）

1. **中等跳空最危险**
   - 3-10x ATR: 50-63% 影响分型
   - >50x ATR: 0% 影响分型（反而安全）

2. **问题合约关联**
   - p2401 分型影响率 82%（最高）
   - p2401 也是唯一 FAIL 的合约
   - 跳空问题可能是 p2401 亏损的重要原因

3. **方向差异不显著**
   - 跳空高开: 54% 影响分型
   - 跳空低开: 58% 影响分型

### 原因分析

极大跳空（>30x ATR）反而安全的原因：
- 大跳空创造明确的价格断层，新结构从零开始
- 不会与旧结构产生"模糊重叠"

中等跳空（3-10x ATR）危险的原因：
- 跳空幅度不够大，无法完全脱离旧结构
- 但又足够大，产生了"伪极值点"
- 包含处理难以正确处理这种模糊情况

## 代码改动（S26b 分级冷却）

```python
# 新参数
gap_extreme_atr: float = 1.5      # 降低阈值，捕获更多中等跳空
gap_cooldown_bars: int = 6        # 默认冷却期延长
gap_tier1_atr: float = 10.0       # <10x ATR: 长冷却(6bar)
gap_tier2_atr: float = 30.0       # <30x: 短冷却(3bar), >30x: 无冷却

# 分级逻辑
if gap_atr >= gap_extreme_atr:
    if gap_atr < gap_tier1_atr:
        cooldown = 6  # 中等跳空最危险
    elif gap_atr < gap_tier2_atr:
        cooldown = 3  # 大跳空次之
    else:
        cooldown = 0  # 极大跳空反而安全
```

## 待验证

手动运行回测（修复 venv 后）：
```powershell
cd E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus
# 先关闭所有 Python 进程
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
# 重新创建 venv
Remove-Item -Recurse -Force .venv
uv sync --extra trade
# 运行回测
uv run python experiments/iter18_gap_fractal/backtest_s26b.py
```

## 预期效果

1. **降低结构性风险**：通过分级冷却减少伪分型导致的错误信号
2. **保持趋势合约收益**：极大跳空不冷却，不伤趋势
3. **改善 p2401**：分型影响率最高的合约应该获益最大

## 子代理状态

- Gemini (gap-fractal-theory): 理论分析中
- GPT (gap-strategy-impact): 策略影响分析中

等待子代理完成后整合结论。
