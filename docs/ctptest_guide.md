# CTP穿透式认证接入指南

本文档说明如何使用 CTPTEST 接口完成期货公司穿透式认证测试，并最终接入实盘交易。

## 零、快速开始

### 已安装依赖

```
vnpy-ctp      6.7.11.2  (实盘交易)
vnpy-ctptest  6.7.2.1   (穿透式测试)
```

### 启动命令

```bash
# 穿透式测试
uv run python -m qp.runtime.trader_app --gateway ctptest

# 实盘交易
uv run python -m qp.runtime.trader_app --gateway ctp

# OpenCTP 7x24仿真
uv run python -m qp.runtime.trader_app --gateway tts
```

### 配置文件

穿透式测试需要创建 `.vntrader/connect_ctptest.json`：

```json
{
    "用户名": "期货账号",
    "密码": "密码",
    "经纪商代码": "期货公司BrokerID",
    "交易服务器": "期货公司提供的测试服务器地址",
    "行情服务器": "期货公司提供的测试行情地址",
    "产品名称": "期货公司提供的AppID",
    "授权编码": "期货公司提供的AuthCode"
}
```

> **注意**: 以上信息需要向期货公司IT部门申请获取。

---

## 一、背景知识

### CTP 与 CTPTEST 的区别

| 接口 | API版本 | 用途 | 数据加密 |
|-----|---------|------|---------|
| **CTPTEST** | 6.7.2 (测试版) | 穿透式认证测试 | 未加密 (便于期货公司审核) |
| **CTP** | 6.7.2 (正式版) | 实盘交易 | 加密传输 |

> **重要**: 所有客户的仿真接入认证测试都**必须**使用 CTPTEST 版本！

### 穿透式监管要求

2019年6月起，中国期货市场实施穿透式监管，要求：
- 所有交易终端必须向期货公司报备
- 采集客户端信息（MAC地址、硬盘序列号、操作系统等）
- 每个交易程序需申请独立的 AppID 和 AuthCode

## 二、安装配置

### 2.1 安装 vnpy_ctptest

项目已在 `pyproject.toml` 中配置依赖，正常情况下已自动安装。

如需手动安装：

```bash
# 使用 uv 安装（推荐）
uv pip install vnpy-ctptest

# 或同步所有trade依赖
uv sync --extra trade

# 或使用 pip
pip install vnpy-ctptest
```

### 2.2 验证安装

```bash
# 检查已安装版本
uv pip list | grep ctp

# 预期输出:
# vnpy-ctp      6.7.11.2
# vnpy-ctptest  6.7.2.1
```

```python
# Python验证
from vnpy_ctptest import CtptestGateway
print("vnpy_ctptest 安装成功")
```

## 三、向期货公司申请接入

### 3.1 准备材料

联系期货公司IT部门，提供以下信息：

1. **个人/机构信息**
   - 期货账号
   - 联系人姓名、电话、邮箱

2. **终端信息**（运行交易程序的电脑）
   - MAC地址
   - 硬盘序列号
   - 操作系统版本

3. **软件信息**
   - 软件名称：QuantPlus
   - 软件版本：0.1.0
   - 开发语言：Python

### 3.2 获取MAC地址

```bash
# Windows
ipconfig /all | findstr "物理地址"

# Linux
ip link show | grep ether
```

### 3.3 获取硬盘序列号

```bash
# Windows (管理员权限)
wmic diskdrive get serialnumber

# Linux
sudo hdparm -I /dev/sda | grep Serial
```

### 3.4 期货公司返回信息

申请通过后，期货公司会提供：

| 参数 | 说明 | 示例 |
|-----|------|------|
| BrokerID | 经纪商代码 | 1234 |
| AppID | 产品名称/应用ID | client_quantplus_1.0 |
| AuthCode | 授权编码 | ABCD1234EFGH5678 |
| 测试交易服务器 | 穿透式测试环境 | tcp://xxx.xxx.xxx.xxx:41205 |
| 测试行情服务器 | 穿透式测试环境 | tcp://xxx.xxx.xxx.xxx:41213 |

## 四、穿透式测试流程

### 4.1 创建测试配置文件

创建 `.vntrader/connect_ctptest.json`：

```json
{
    "用户名": "你的期货账号",
    "密码": "你的密码",
    "经纪商代码": "期货公司提供的BrokerID",
    "交易服务器": "期货公司提供的测试交易服务器",
    "行情服务器": "期货公司提供的测试行情服务器",
    "产品名称": "期货公司提供的AppID",
    "授权编码": "期货公司提供的AuthCode"
}
```

### 4.2 修改 trader_app.py 支持 CTPTEST

项目已支持通过命令行参数切换网关，需要添加 ctptest 选项：

```python
# src/qp/runtime/trader_app.py

parser.add_argument(
    "--gateway",
    type=str,
    choices=["ctp", "tts", "ctptest"],  # 添加 ctptest
    default="ctp",
    help="交易网关: ctp=实盘, tts=OpenCTP, ctptest=穿透式测试",
)

# 在 main() 中添加 ctptest 分支
if args.gateway == "ctptest":
    from vnpy_ctptest import CtptestGateway
    gateway_cls = CtptestGateway
    gateway_name = "CTPTEST (穿透式测试)"
```

### 4.3 启动穿透式测试

```bash
# 使用 CTPTEST 网关启动
uv run python -m qp.runtime.trader_app --gateway ctptest
```

### 4.4 执行测试

1. 在 GUI 中点击 **系统 → 连接CTPTEST**
2. 输入期货公司提供的测试账号信息
3. 点击连接

### 4.5 验证测试成功

当看到以下日志时，表示测试成功：

```
CTP交易接口初始化成功
CTP行情接口初始化成功
合约信息查询成功
```

### 4.6 通知期货公司

将测试成功的截图发送给期货公司IT部门，等待审核。

## 五、切换到实盘

### 5.1 期货公司审核通过后

期货公司会：
1. 将你的 AppID/AuthCode 添加到实盘服务器
2. 提供实盘服务器地址

### 5.2 创建实盘配置文件

创建 `.vntrader/connect_ctp.json`：

```json
{
    "用户名": "你的期货账号",
    "密码": "你的密码",
    "经纪商代码": "期货公司BrokerID",
    "交易服务器": "期货公司实盘交易服务器",
    "行情服务器": "期货公司实盘行情服务器",
    "产品名称": "你的AppID",
    "授权编码": "你的AuthCode"
}
```

### 5.3 启动实盘交易

```bash
# 使用 CTP 网关启动（实盘）
uv run python -m qp.runtime.trader_app --gateway ctp
```

## 六、常见问题

### Q1: CTP和CTPTEST能同时加载吗？

**不能！** 两个接口会冲突，只能加载其中一个。

### Q2: 测试必须用申报的电脑吗？

**是的！** 期货公司会验证MAC地址，必须使用申报时填写的那台电脑。

### Q3: 报错 "CTP不合法的登录"？

检查：
1. 是否使用了正确的接口（CTPTEST vs CTP）
2. AppID 和 AuthCode 是否正确
3. 服务器地址是否正确

### Q4: 报错 "客户端认证失败"？

可能原因：
1. MAC地址与申报不符
2. AppID/AuthCode 未在服务器端注册
3. 使用了错误的API版本

### Q5: SimNow 和穿透式测试的区别？

| 环境 | 用途 | 需要认证 |
|-----|------|---------|
| SimNow | 开发调试、策略回测 | 不需要 |
| 穿透式测试 | 期货公司认证 | 需要 |
| 实盘 | 真实交易 | 需要 |

## 七、网关对比

| 网关 | 用途 | 配置文件 |
|-----|------|---------|
| `ctp` | 实盘交易 / SimNow | connect_ctp.json |
| `ctptest` | 穿透式认证测试 | connect_ctptest.json |
| `tts` | OpenCTP 7x24仿真 | connect_tts.json |

## 八、参考资料

- [VeighNa官方文档 - 交易接口](https://www.vnpy.com/docs/cn/community/info/gateway.html)
- [彻底搞定期货穿透式CTP API接入](https://www.vnpy.com/forum/topic/603-kan-wan-zhe-pian-che-di-gao-ding-qi-huo-chuan-tou-shi-ctp-apijie-ru)
- [vnpy_ctptest GitHub](https://github.com/vnpy/vnpy_ctptest)
- [vnpy_ctptest PyPI](https://pypi.org/project/vnpy-ctptest/)
