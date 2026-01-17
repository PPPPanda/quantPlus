# Data 目录完整说明

## 目录概述

`data/` 目录存储所有量化交易相关的历史数据、实时数据缓存和数据源配置。

```
data/
├── datadir/          # 迅投研数据目录（主要数据源）
├── openbb/           # OpenBB数据缓存
├── data/             # 其他数据源
├── quoter/           # 行情服务数据
└── log/              # 数据获取日志
```

---

## 1. datadir/ - 迅投研数据目录

这是**主要数据存储目录**，包含从迅投研(XTQuant)获取的期货、股票、期权历史数据。

### 1.1 交易所数据目录

每个交易所有独立的数据目录：

#### 期货交易所

| 目录 | 交易所全称 | 品种示例 | 说明 |
|------|-----------|---------|------|
| **DF/** | 大连商品交易所<br>Dalian Commodity Exchange | 豆粕(m)、棕榈油(p)、铁矿石(i)、焦炭(j)、豆油(y)、玉米(c) | 农产品、黑色系 |
| **SF/** | 上海期货交易所<br>Shanghai Futures Exchange | 铜(cu)、铝(al)、锌(zn)、黄金(au)、白银(ag)、螺纹钢(rb)、橡胶(ru) | 有色金属、贵金属 |
| **ZF/** | 郑州商品交易所<br>Zhengzhou Commodity Exchange | 白糖(SR)、棉花(CF)、PTA(TA)、甲醇(MA)、菜油(OI)、苹果(AP) | 农产品、化工 |
| **GF/** | 广州期货交易所<br>Guangzhou Futures Exchange | 工业硅(si)、碳酸锂(lc)、PTA(pd) | 新能源材料 |
| **INE/** | 上海国际能源交易中心<br>Shanghai INE | 原油(sc) | 能源期货 |
| **IF/** | 中国金融期货交易所<br>China Financial Futures Exchange | 沪深300(IF)、上证50(IH)、中证500(IC)、国债期货(T/TF) | 金融期货 |

#### 证券交易所

| 目录 | 交易所 | 说明 |
|------|--------|------|
| **SH/** | 上海证券交易所 | A股、ETF、债券 |
| **SZ/** | 深圳证券交易所 | A股、ETF、创业板 |
| **SHO/** | 上交所期权 | 50ETF期权、300ETF期权 |
| **SZO/** | 深交所期权 | 创业板ETF期权 |

#### 其他

| 目录 | 说明 |
|------|------|
| **EP/** | 欧洲期货交易所（European Exchange） |
| **BJ/** | 北京证券交易所（原油、LPG等） |
| **HGT/** | 沪港通 |
| **SGT/** | 深港通 |

---

### 1.2 数据文件结构

每个交易所目录下包含多个周期的数据：

```
DF/                    # 大连商品交易所
├── 60/                # 1分钟数据
│   ├── p00.DAT        # 棕榈油主力合约分钟数据
│   └── y00.DAT        # 豆油主力合约分钟数据
├── 300/               # 5分钟数据（300秒）
├── 900/               # 15分钟数据（900秒）
├── 1800/              # 30分钟数据
├── 3600/              # 1小时数据
└── 86400/             # 日线数据（86400秒=1天）
    ├── a_9502.bo      # 豆一2025年2月合约
    ├── p_9502.bo      # 棕榈油2025年2月合约
    └── ...
```

#### 文件扩展名说明

| 扩展名 | 数据类型 | 结构 |
|--------|---------|------|
| **.DAT** | 主力合约连续数据 | 二进制，包含 OHLCV（开高低收量） |
| **.bo** | 单个合约历史数据 | Binary OHLC，固定长度记录 |
| **.fe** | 合约列表索引 | 存储历史合约代码和到期日 |

#### 合约命名规则

```
p_9502.bo
│ │  │
│ │  └─ 02: 2月份
│ └──── 95: 2025年
└────── p:  棕榈油品种代码

主力合约：p00.DAT（00表示主力）
```

**品种代码示例**：
- `p` = 棕榈油 (Palm Oil)
- `m` = 豆粕 (soybean Meal)
- `y` = 豆油 (soybean oil)
- `i` = 铁矿石 (Iron ore)
- `rb` = 螺纹钢 (Rebar)
- `cu` = 铜 (Copper)
- `au` = 黄金 (Gold)

---

### 1.3 历史合约索引

```
historycontracts_auto/
└── INE.fe           # 上海能源中心历史合约列表
```

**.fe 文件**：
- 存储每个交易所的所有历史合约代码
- 包含合约到期日、上市日期
- 用于数据查询时定位正确的合约文件

**示例内容**：
```
INE.fe:
sc2001, 2020-01-15  # 原油2020年1月合约
sc2002, 2020-02-15  # 原油2020年2月合约
...
```

---

### 1.4 增量索引文件

```
increase/
├── DF               # 大连商品交易所增量索引
├── SF               # 上海期货交易所增量索引
├── ZF               # 郑州商品交易所增量索引
├── SH               # 上海证券交易所增量索引
└── ...
```

**用途**：
- 记录每个合约数据文件的**最后更新位置**
- 支持增量数据下载（只下载新数据，不重复下载）
- 二进制格式，包含文件偏移量、时间戳

**运行时生成**，不应提交到Git。

---

### 1.5 交易时间配置

```
quotetimeinfo        # 各品种交易时间配置
```

**用途**：
- 存储每个期货品种的交易时段
- 区分日盘、夜盘时间
- 用于数据获取和策略回测时间过滤

**示例**（棕榈油）：
```
p (棕榈油):
  日盘: 09:00-10:15, 10:30-11:30, 13:30-15:00
  夜盘: 21:00-23:00
```

---

### 1.6 板块数据

```
Sector/              # 行业板块分类
SectorData/          # 板块成分股数据
```

**用途**：
- 股票行业分类（金融、科技、消费等）
- 板块成分股列表
- 板块指数数据

---

## 2. openbb/ - OpenBB 数据缓存

```
openbb/
└── *.csv            # OpenBB数据源缓存
```

**OpenBB**：开源金融数据平台
- 提供股票、加密货币、宏观经济数据
- CSV格式缓存，避免重复API调用
- 数据更新周期：按需更新

**已被gitignore**，不提交到版本控制。

---

## 3. data/ - 其他数据源

```
data/
└── (待补充的其他数据源)
```

保留目录，用于扩展其他数据源。

---

## 4. quoter/ - 行情服务数据

```
quoter/
└── (行情服务配置和缓存)
```

实时行情推送服务的配置和数据缓存。

---

## 5. log/ - 数据获取日志

```
log/
└── *.log            # 数据下载、更新日志
```

记录数据源连接、下载状态、错误信息。

---

## 数据使用示例

### 获取棕榈油分钟数据

```python
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest
from vnpy_xt import Datafeed as XtDatafeed

datafeed = XtDatafeed()
datafeed.init()

req = HistoryRequest(
    symbol="p00",           # 棕榈油主力合约
    exchange=Exchange.DCE,   # 大连商品交易所
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    interval=Interval.MINUTE  # 1分钟K线
)

bars = datafeed.query_bar_history(req)
# 数据来源: data/datadir/DF/60/p00.DAT
```

### 数据文件路径规则

```python
# 路径生成逻辑：
# data/datadir/{交易所}/{周期秒数}/{合约代码}.DAT

交易所代码映射：
  Exchange.DCE  → DF (大连)
  Exchange.SHFE → SF (上海期货)
  Exchange.CZCE → ZF (郑州)
  Exchange.CFFEX → IF (中金所)
  Exchange.INE  → INE (能源中心)

周期映射：
  Interval.MINUTE → 60
  Interval.HOUR   → 3600
  Interval.DAILY  → 86400
```

---

## 数据更新策略

### 自动更新

迅投研数据在策略运行时**自动增量更新**：
1. 检查 `increase/` 索引文件
2. 下载新增数据追加到 `.DAT` / `.bo` 文件
3. 更新索引偏移量

### 手动更新

```python
# 通过vnpy_xt接口自动触发
datafeed = XtDatafeed()
datafeed.init()  # 连接迅投研
datafeed.query_bar_history(req)  # 自动更新数据
```

---

## Git 管理策略

### 应该提交的文件

```bash
# 仅提交.gitkeep占位文件
data/datadir/.gitkeep
data/openbb/.gitkeep
```

### 应该忽略的文件

```gitignore
# 所有运行时数据
data/datadir/**/*.DAT
data/datadir/**/*.bo
data/datadir/**/*.fe
data/datadir/increase/*      # 增量索引（运行时生成）
data/openbb/*.csv             # OpenBB缓存
data/log/                     # 日志
```

**原因**：
- ✅ 数据文件体积大（GB级）
- ✅ 可从数据源重新获取
- ✅ 每次运行都会改变
- ✅ 不是源代码的一部分

---

## 数据目录大小参考

| 目录 | 典型大小 | 说明 |
|------|---------|------|
| `DF/60/` | 5-10 MB/合约 | 1年分钟数据 ≈ 10万根K线 |
| `DF/86400/` | 50-100 KB/合约 | 5年日线数据 ≈ 1200根K线 |
| `increase/DF` | 5-6 MB | 索引所有大连品种 |
| `总计` | **30-50 GB** | 完整多品种多周期数据 |

---

## 常见问题

### Q: 为什么.bo文件显示为"OpenPGP Public Key"？

A: 这是`file`命令的误判，`.bo`实际上是**二进制OHLC数据**，不是PGP密钥。

### Q: 如何清理旧数据？

A: 删除整个 `data/datadir/` 目录，下次运行时会自动重新下载。

```bash
rm -rf data/datadir/DF/
# 下次运行策略时自动重建
```

### Q: 数据不同步怎么办？

A: 删除对应的 `increase/` 索引文件，强制全量更新。

```bash
rm data/datadir/increase/DF
# 下次会全量扫描并更新索引
```

---

## 扩展阅读

- [迅投研官方文档](https://dict.thinktrader.net/)
- [VeighNa数据接口文档](https://www.vnpy.com/docs/cn/datafeed.html)
- [OpenBB官方网站](https://openbb.co/)

---

*文档更新时间：2026-01-17*
