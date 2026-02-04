# iter8 Phase 1: 2B背驰重写设计（Claude方案）

## 核心思路

### 走势段(Segment)定义
从 `_bi_points` 中提取同向走势段：
- **下降段**：从一个top开始，连续笔低点依次走低，直到出现底分型反转
  - 例：top(100) → bot(90) → top(95) → bot(85) → 这是一个下降段
  - 段力度 = 该段内所有笔的MACD histogram面积之和
- **上升段**：从一个bottom开始，连续笔高点依次走高

### 背驰2B信号条件（底背驰→买入）
1. 存在至少2个下降段（当前段 + 前一个同向段）
2. 当前下降段创新低（最低点 < 前一段最低点）
3. 当前下降段MACD面积 < 前一段MACD面积 × threshold
4. 当前段结束（新bottom形成后出现向上笔确认）
5. 15m趋势过滤 is_bull
6. S2b: 中枢存在(forming/active)

### 实现方案

不需要完整的 segment 数据结构，可以简化：
- 利用已有的 `_bi_macd_areas` 列表
- 比较最近2-3笔的累积面积 vs 之前2-3笔的累积面积
- 关键：背驰 = 价格创新低/新高但MACD力度衰减

### 简化版实现（低改动，渐进式）

**Phase A: 修复2B的"diff增强"为"diff衰减"**
- 当前：`diff_ok = p_now.diff > p_prev.diff`（动能增强）
- 修改：`diff_ok = abs(p_now.diff) < abs(p_prev.diff)`（动能衰减 = 背驰）
- 这只是方向修正，风险最低

**Phase B: 加入面积背驰作为必要条件**
- 当前面积背驰是OR条件（div_mode=1），改为AND或替换
- 用 `_bi_macd_areas[-1] < _bi_macd_areas[-3] * 0.7` 作为必要条件

**Phase C: 引入多笔累积面积比较**
- 累积最近N笔同向面积 vs 之前N笔同向面积
- 更接近走势段力度比较
