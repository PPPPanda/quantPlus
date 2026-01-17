# OpenCTP 集成技术调研报告

**调研日期**: 2026-01-17
**调研目标**: 为 QuantPlus 集成 OpenCTP 7x24 模拟交易环境
**参考资源**:
- [OpenCTP GitHub](https://github.com/openctp/openctp)
- [OpenCTP 官网](http://www.openctp.cn/)
- [VeighNa TTS 接口](https://github.com/vnpy/vnpy_tts)
- [VeighNa 社区讨论](https://www.vnpy.com/forum/topic/30696)

---

## 一、OpenCTP 技术架构

### 1.1 核心功能

**OpenCTP** 是一个 CTP 生态技术服务平台，提供：

| 功能模块 | 说明 |
|---------|------|
| **多柜台兼容** | 支持华鑫 TORA、中泰 XTP、东方财富 EMT 等多个证券柜台的 CTPAPI 兼容接口 |
| **TTS 模拟环境** | 基于 TTS（Tick Trading System）的 7x24 模拟交易系统 |
| **全品种支持** | 期货、期权、A股、基金、债券、港美股模拟交易 |
| **SimNow 替代** | 更稳定的模拟交易环境（SimNow 停服频繁） |

### 1.2 TTS 系统特点

```
TTS (Tick Trading System)
├── 7x24 环境 - 轮播历史行情，全天候可用
├── 仿真环境 - 与实盘交易时段一致
├── VIP 环境 - 高级功能（需单独申请）
└── CTPAPI 兼容 - 使用 CTP 标准接口
```

**关键优势**：
- ✅ 7x24 小时可用（SimNow 经常停服）
- ✅ 支持全品种模拟交易
- ✅ CTPAPI 兼容，无需修改代码
- ✅ 免费账号申请

---

## 二、VeighNa 集成方案对比

### 2.1 方案 A：使用 vnpy_tts（推荐）

**特点**：
- VeighNa 官方维护的 TTS 接口包
- 基于 TTS 6.7.2 API
- 无需手动替换 DLL
- Python 3.10-3.13 支持

**安装方式**：
```bash
pip install vnpy_tts
```

**代码示例**：
```python
from vnpy_tts import TtsGateway

main_engine.add_gateway(TtsGateway)
```

**优点**：
- ✅ 官方支持，更新及时
- ✅ 安装简单，无需手动操作 DLL
- ✅ 与 vnpy 生态完全兼容
- ✅ 独立包，不影响现有 CTP 配置

**缺点**：
- ⚠️ 只支持 TTS 环境，不支持其他柜台
- ⚠️ 版本较旧（6.7.2 vs 当前 CTP 6.7.11）

---

### 2.2 方案 B：替换 vnpy_ctp 的 DLL

**特点**：
- 使用 OpenCTP 提供的 TTS-CTPAPI DLL 替换原 CTP DLL
- 保持 vnpy_ctp 接口不变
- 需要版本匹配（当前 vnpy_ctp 6.7.11.2）

**操作步骤**（传统方法）：
```bash
# 1. 定位 DLL 位置
cd .venv/Lib/site-packages/vnpy_ctp/api/

# 2. 备份原 DLL
cp thosttraderapi_se.dll thosttraderapi_se.dll.backup
cp thostmduserapi_se.dll thostmduserapi_se.dll.backup

# 3. 下载 OpenCTP TTS-CTPAPI DLL（需匹配 6.7.11 版本）
# 从 openctp-ctp PyPI 包或 GitHub releases

# 4. 替换 DLL
# 将下载的 DLL 覆盖原文件
```

**优点**：
- ✅ 可使用最新版本的 API
- ✅ 无需修改现有代码

**缺点**：
- ❌ 手动操作复杂，易出错
- ❌ 版本匹配困难（OpenCTP TTS 最新为 6.7.2，vnpy_ctp 已到 6.7.11）
- ❌ 更新 vnpy_ctp 时需重新替换
- ❌ **VeighNa 2.5+ 已明确说明不需要此方法**

---

### 2.3 方案 C：使用 openctp-ctp PyPI 包（不推荐）

**特点**：
- OpenCTP 官方提供的 Python 包
- 版本：openctp-ctp 6.7.11.0 / openctp-tts 6.7.2.x

**问题**：
- ❌ 与 vnpy_ctp 包冲突（同名 DLL）
- ❌ VeighNa 未官方支持
- ❌ 需要修改 vnpy 源码

**结论**: 不适用于 VeighNa 框架

---

## 三、关键技术细节

### 3.1 DLL 同名冲突问题

**重要**：TTS 和 CTP 的 DLL 文件同名：
```
thosttraderapi_se.dll
thostmduserapi_se.dll
```

**后果**：
- 在同一进程中同时加载会导致 **错误 4097**（连接失败）
- VeighNa GUI 中 **不能同时勾选 CTP 和 TTS**

**解决方案**：
1. 使用独立的网关包（vnpy_tts vs vnpy_ctp）
2. 运行时只加载其中一个
3. 或使用虚拟环境隔离

### 3.2 版本兼容性矩阵

| 包名 | API 版本 | Python | 平台 | 维护状态 |
|------|---------|--------|------|---------|
| vnpy_ctp | 6.7.11 | 3.10-3.13 | Win/Linux/Mac | ✅ 官方维护 |
| vnpy_tts | 6.7.2 | 3.10-3.13 | Win/Linux | ✅ 官方维护 |
| openctp-ctp | 6.7.11.0 | 3.8-3.14 | Win/Linux | ⚠️ OpenCTP 维护 |
| openctp-tts | 6.7.2.x | 3.8-3.14 | Win/Linux | ⚠️ OpenCTP 维护 |

**当前环境**：
- Python: 3.13
- vnpy_ctp: 6.7.11.2
- 操作系统: Windows 10

### 3.3 服务器地址配置

#### OpenCTP TTS 7x24 环境（用户已配置）

```json
{
    "用户名": "16714",
    "密码": "123456",
    "经纪商代码": "9999",
    "交易服务器": "tcp://trading.openctp.cn:30001",
    "行情服务器": "tcp://trading.openctp.cn:30011",
    "产品名称": "",
    "授权编码": "",
    "柜台环境": "实盘"
}
```

**注意**：这些地址是 TTS 环境，**需要使用 TTS-CTPAPI** 才能连接。

#### 其他可用地址

**7x24 环境（历史数据轮播）**：
```
交易: tcp://122.51.136.165:20002
行情: tcp://122.51.136.165:20004
```

**仿真环境（交易时段）**：
```
交易: tcp://121.36.146.182:20002
行情: tcp://121.36.146.182:20004
```

---

## 四、推荐实施方案

### 4.1 最佳方案：双网关支持

**目标**：在 QuantPlus 中同时支持 CTP 和 TTS，用户可选择

**实施步骤**：

#### 步骤 1：安装 vnpy_tts
```bash
uv pip install vnpy_tts
```

#### 步骤 2：修改 trader_app.py 支持网关选择

```python
# src/qp/runtime/trader_app.py

def create_main_engine(gateway_name: str = "CTP") -> MainEngine:
    """
    创建主引擎实例

    Args:
        gateway_name: 网关名称，支持 "CTP" 或 "TTS"
    """
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 根据参数选择网关
    if gateway_name.upper() == "TTS":
        from vnpy_tts import TtsGateway
        main_engine.add_gateway(TtsGateway)
    else:  # 默认 CTP
        from vnpy_ctp import CtpGateway
        main_engine.add_gateway(CtpGateway)

    return main_engine, event_engine
```

#### 步骤 3：添加命令行参数

```python
parser.add_argument(
    "--gateway",
    type=str,
    choices=["ctp", "tts"],
    default="ctp",
    help="选择交易网关: ctp（SimNow/CTP实盘）或 tts（OpenCTP 7x24）"
)
```

#### 步骤 4：创建 TTS 配置文件

```bash
# .vntrader/connect_tts.json
{
    "用户名": "16714",
    "密码": "123456",
    "经纪商代码": "9999",
    "交易服务器": "tcp://trading.openctp.cn:30001",
    "行情服务器": "tcp://trading.openctp.cn:30011",
    "产品名称": "",
    "授权编码": ""
}
```

#### 步骤 5：更新文档和测试

**使用方式**：
```bash
# 使用 CTP（SimNow）
uv run python -m qp.runtime.trader_app --gateway ctp

# 使用 TTS（OpenCTP 7x24）
uv run python -m qp.runtime.trader_app --gateway tts
```

---

### 4.2 方案优势

| 特性 | 说明 |
|------|------|
| **兼容性** | 保持现有 CTP 功能不变 |
| **灵活性** | 用户可根据需求选择网关 |
| **稳定性** | TTS 7x24 可用，不受 SimNow 停服影响 |
| **可维护性** | 使用官方包，自动更新 |
| **隔离性** | 两个网关独立，不会冲突 |

---

## 五、常见问题与解决方案

### Q1: 连接 OpenCTP 报错 4097

**原因**：
1. 同时加载了 CTP 和 TTS 网关（DLL 冲突）
2. 使用 CTP DLL 连接 TTS 服务器（API 不匹配）
3. 网络问题或服务器地址错误

**解决**：
```python
# 确保只加载一个网关
# ❌ 错误示例
main_engine.add_gateway(CtpGateway)
main_engine.add_gateway(TtsGateway)  # 冲突！

# ✅ 正确示例
if use_tts:
    main_engine.add_gateway(TtsGateway)
else:
    main_engine.add_gateway(CtpGateway)
```

### Q2: vnpy_tts 版本较旧（6.7.2），会有问题吗？

**答案**：不会

- TTS 服务器使用 6.7.2 API
- vnpy_tts 6.7.2.3 已经是匹配的稳定版本
- 期货交易的核心接口自 6.3.15 以来变化不大
- VeighNa 官方持续维护该包

### Q3: 是否需要手动替换 DLL？

**答案**：不需要

- VeighNa 2.5+ 已内置支持
- vnpy_tts 包含正确的 TTS DLL
- 手动替换反而可能引入版本不匹配问题

### Q4: OpenCTP 账号如何申请？

**申请方式**：
1. 访问 [OpenCTP GitHub](https://github.com/krenx1983/openctp)
2. 根据说明提交申请
3. 获取 7x24 测试账号

**当前账号**：
- 用户名：16714
- 密码：123456
- 类型：7x24 模拟账号

### Q5: TTS 是否提供历史数据下载？

**答案**：否

OpenCTP 只是仿真环境，不提供历史数据服务。历史数据需要：
- 使用 vnpy_datamanager 从其他源下载
- 或使用已集成的迅投研数据接口

---

## 六、实施风险评估

### 6.1 技术风险

| 风险项 | 风险等级 | 缓解措施 |
|--------|---------|---------|
| DLL 冲突导致系统崩溃 | 🟡 中 | 使用独立网关包，不同时加载 |
| API 版本不兼容 | 🟢 低 | 使用官方 vnpy_tts，版本已匹配 |
| 网络连接不稳定 | 🟡 中 | 实现重连机制，监控连接状态 |
| 配置文件混淆 | 🟢 低 | 使用不同文件名（connect_ctp vs connect_tts） |

### 6.2 业务风险

| 风险项 | 风险等级 | 说明 |
|--------|---------|------|
| 模拟环境数据准确性 | 🟡 中 | TTS 使用历史数据轮播，非实时 |
| 账号失效 | 🟢 低 | 免费账号，可重新申请 |
| 服务可用性 | 🟢 低 | 7x24 环境，比 SimNow 更稳定 |

---

## 七、下一步行动计划

### 阶段 1：环境准备（预计 30 分钟）
- [x] 完成技术调研
- [ ] 安装 vnpy_tts 包
- [ ] 验证包安装成功

### 阶段 2：代码集成（预计 1 小时）
- [ ] 修改 trader_app.py 支持网关选择
- [ ] 添加命令行参数解析
- [ ] 创建 TTS 配置文件模板
- [ ] 更新配置加载逻辑

### 阶段 3：测试验证（预计 30 分钟）
- [ ] 测试 CTP 网关功能（确保不影响现有功能）
- [ ] 测试 TTS 网关连接
- [ ] 验证合约信息获取
- [ ] 测试行情订阅

### 阶段 4：文档完善（预计 30 分钟）
- [ ] 创建 OpenCTP 使用指南
- [ ] 更新 README
- [ ] 添加配置示例
- [ ] 记录常见问题

---

## 八、参考资源

### 官方文档
- [OpenCTP GitHub](https://github.com/openctp/openctp) - OpenCTP 源码和文档
- [OpenCTP 官网](http://www.openctp.cn/) - 服务器地址和申请
- [VeighNa TTS 接口](https://github.com/vnpy/vnpy_tts) - 官方 TTS 网关
- [VeighNa 文档](https://www.vnpy.com/docs) - VeighNa 框架文档

### 社区讨论
- [vn.py 连接 TTS 通道注意事项](https://www.vnpy.com/forum/topic/30696) - 官方使用说明
- [SimNow 备胎 OpenCTP 使用说明](https://www.vnpy.com/forum/topic/31071) - 社区实践经验
- [CTP 接口量化交易资料汇总](https://www.vnpy.com/forum/topic/31506) - 综合资源

### PyPI 包
- [vnpy-tts](https://pypi.org/project/vnpy-tts/) - VeighNa TTS 网关
- [openctp-ctp](https://pypi.org/project/openctp-ctp/) - OpenCTP CTP 包
- [openctp-tts](https://pypi.org/project/openctp-tts/) - OpenCTP TTS 包

---

## 九、结论与建议

### 核心结论

1. **推荐使用 vnpy_tts 包**
   - 官方支持，稳定可靠
   - 安装简单，无需手动操作
   - 与现有 vnpy_ctp 互不干扰

2. **不推荐手动替换 DLL**
   - VeighNa 2.5+ 已弃用此方法
   - 版本匹配复杂，易出错
   - 维护成本高

3. **双网关方案最优**
   - 保留 CTP（SimNow/实盘）
   - 新增 TTS（7x24 测试）
   - 用户灵活选择

### 实施建议

**立即执行**：
- 安装 vnpy_tts
- 实现网关选择功能
- 测试 OpenCTP 连接

**后续优化**：
- 添加网关状态监控
- 实现自动重连
- 完善错误处理

**文档更新**：
- 创建用户指南
- 更新快速开始教程
- 记录故障排查步骤

---

*调研完成时间：2026-01-17*
*下一步：开始实施阶段 1 - 环境准备*
