# OpenCTP 快速上手指南

**OpenCTP** 是一个提供 7x24 小时模拟交易环境的CTP兼容接口，可作为 SimNow 的稳定替代方案。

---

## 快速开始

### 1. 测试连接（推荐）

在启动 GUI 之前，先测试连接是否正常：

```bash
cd /e/work/quant/quantPlus
uv run python tests/test_openctp_connection.py
```

**预期输出**：
```
============================================================
OpenCTP TTS 网关连接测试
============================================================
[OK] 加载配置: 用户名=16714, 交易服务器=tcp://trading.openctp.cn:30001

[步骤 1] 创建事件引擎和主引擎...
[OK] 引擎创建成功

[步骤 2] 添加 TTS 网关...
[OK] TTS 网关添加成功

[步骤 3] 连接到 OpenCTP TTS 服务器...
[日志] 交易前置服务器连接成功
[日志] 行情服务器连接成功
[日志] 交易前置服务器登录成功
[日志] 投资者信息确认成功
[日志] 行情服务器登录成功

[OK] 连接成功！已接收合约信息（耗时 1 秒）
[OK] 共获取 5000+ 个合约
```

### 2. 启动 GUI

使用 TTS 网关启动交易界面：

```bash
# 使用 OpenCTP TTS (7x24 模拟环境)
uv run python -m qp.runtime.trader_app --gateway tts

# 或继续使用 CTP (SimNow)
uv run python -m qp.runtime.trader_app --gateway ctp
```

### 3. 连接服务器

1. 在 GUI 中点击 **系统 → 连接TTS**
2. 等待状态栏显示 "已连接"
3. 可以开始查看合约、订阅行情

---

## 账号配置

### 配置文件位置

```
.vntrader/connect_tts.json
```

### 默认配置

```json
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

### 账号信息

- **用户名**: `16714`
- **密码**: `123456`
- **类型**: 7x24 模拟账号
- **申请**: 通过 [OpenCTP GitHub](https://github.com/krenx1983/openctp)

---

## 支持的功能

### ✅ 全品种交易

| 品种类别 | 示例合约 | 说明 |
|---------|---------|------|
| **商品期货** | p2505, rb2505, i2505 | 棕榈油、螺纹钢、铁矿石等 |
| **金融期货** | IF2601, IC2601, IH2601 | 沪深300、中证500、上证50 |
| **期权** | IO2601-C-4000 | 股指期权 |
| **A股** | 600000.SH, 000001.SZ | 股票模拟交易 |

### ✅ 7x24 可用

- **交易时间**: 全天候，不受实盘交易时段限制
- **行情数据**: 历史数据轮播，持续更新
- **账号有效期**: 长期有效（建议定期确认）

### ✅ CTPAPI 兼容

- 与 vnpy_ctp 接口完全兼容
- 无需修改策略代码
- 可无缝切换 CTP/TTS

---

## 使用场景

### 场景 1：策略开发测试

```bash
# 使用 OpenCTP 进行策略测试
uv run python -m qp.runtime.trader_app --gateway tts --profile all

# 在 GUI 中:
# 1. 连接 TTS
# 2. 功能 → CTA策略
# 3. 加载策略并测试
```

### 场景 2：7x24 回测验证

```python
# 在任何时间都可以连接获取数据
# 不受 SimNow 停服影响
from vnpy_tts import TtsGateway
main_engine.add_gateway(TtsGateway)
main_engine.connect(tts_config, "TTS")
```

### 场景 3：教学演示

```bash
# 适合课堂演示、培训使用
# 不需要等待特定交易时段
uv run python -m qp.runtime.trader_app --gateway tts
```

---

## 网关切换

### CTP vs TTS 对比

| 特性 | CTP (SimNow) | TTS (OpenCTP) |
|------|-------------|---------------|
| **服务时间** | 交易时段 | 7x24 小时 |
| **稳定性** | 经常停服维护 | 持续可用 |
| **数据类型** | 实时行情 | 历史轮播行情 |
| **品种支持** | 期货 + 期权 | 期货 + 期权 + 股票 |
| **账号申请** | 实名注册 | GitHub 申请 |

### 切换方法

```bash
# 方法 1：命令行参数
uv run python -m qp.runtime.trader_app --gateway tts  # 使用 TTS
uv run python -m qp.runtime.trader_app --gateway ctp  # 使用 CTP

# 方法 2：在 GUI 中手动选择
# 系统 → 连接TTS  或  系统 → 连接CTP
```

**重要**：同一时间只能连接一个网关（CTP 和 TTS 的 DLL 同名，会冲突）

---

## 常见问题

### Q1: 连接失败，报错 4097

**原因**：
1. 同时运行了其他使用 CTP DLL 的程序
2. GUI 中同时勾选了 CTP 和 TTS

**解决**：
1. 关闭其他 CTP 程序
2. 只勾选 TTS 网关
3. 重启 QuantPlus

### Q2: 获取不到合约信息

**原因**：
1. 网络连接问题
2. 服务器地址错误
3. 用户名密码错误

**解决**：
1. 检查配置文件 `.vntrader/connect_tts.json`
2. 确认用户名为 `16714`，密码为 `123456`
3. 测试网络连接：`ping trading.openctp.cn`

### Q3: 行情数据不更新

**说明**：
- OpenCTP TTS 使用历史数据轮播
- 不是实时行情，而是模拟行情
- 适合策略测试，不适合实盘决策

### Q4: 与 SimNow 有什么区别？

| 项目 | SimNow | OpenCTP TTS |
|------|--------|-------------|
| 行情类型 | 实时 | 历史轮播 |
| 可用时间 | 交易时段 | 7x24 小时 |
| 停服频率 | 频繁 | 很少 |
| 品种覆盖 | 期货 | 期货+股票+期权 |

---

## 高级配置

### 自定义服务器地址

除了默认的 7x24 环境，还可以使用其他环境：

```json
{
    "用户名": "16714",
    "密码": "123456",
    "经纪商代码": "9999",
    "交易服务器": "tcp://122.51.136.165:20002",
    "行情服务器": "tcp://122.51.136.165:20004",
    "产品名称": "",
    "授权编码": ""
}
```

**可用环境**：

| 环境 | 交易前置 | 行情前置 | 特点 |
|------|---------|---------|------|
| **7x24** | tcp://122.51.136.165:20002 | tcp://122.51.136.165:20004 | 轮播行情 |
| **仿真** | tcp://121.36.146.182:20002 | tcp://121.36.146.182:20004 | 交易时段 |
| **官方** | tcp://trading.openctp.cn:30001 | tcp://trading.openctp.cn:30011 | 推荐使用 |

---

## 技术支持

### 官方资源

- [OpenCTP GitHub](https://github.com/openctp/openctp) - 源码和文档
- [OpenCTP 官网](http://www.openctp.cn/) - 服务器地址和公告
- [VeighNa TTS 接口](https://github.com/vnpy/vnpy_tts) - 官方网关
- [VeighNa 社区](https://www.vnpy.com/forum/) - 社区讨论

### 问题反馈

如遇到问题，请提供以下信息：

1. **错误日志**：`.vntrader/log/` 目录下的最新日志
2. **配置文件**：`.vntrader/connect_tts.json`（隐藏密码）
3. **测试输出**：`uv run python tests/test_openctp_connection.py`
4. **环境信息**：
   - Python 版本：`python --version`
   - vnpy_tts 版本：`pip show vnpy-tts`
   - 操作系统：Windows/Linux/Mac

---

## 下一步

1. **测试连接**：运行测试脚本确认连接正常
2. **查看合约**：在 GUI 中浏览可用合约
3. **订阅行情**：测试行情订阅功能
4. **运行策略**：使用 CTA策略 模块测试策略
5. **数据录制**：使用 DataRecorder 录制行情数据

---

## 附录：测试脚本说明

测试脚本位置：`tests/test_openctp_connection.py`

**测试内容**：
1. ✅ 加载配置文件
2. ✅ 创建主引擎
3. ✅ 添加 TTS 网关
4. ✅ 连接服务器
5. ✅ 登录验证
6. ✅ 获取合约信息
7. ✅ 订阅行情数据
8. ✅ 断开连接

**运行方式**：
```bash
uv run python tests/test_openctp_connection.py
```

**典型耗时**：
- 连接建立：1-3 秒
- 合约下载：2-5 秒
- 行情订阅：立即
- 总计：约 10 秒

---

*文档更新时间：2026-01-17*
*版本：1.0.0*
