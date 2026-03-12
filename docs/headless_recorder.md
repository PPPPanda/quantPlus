# 无头采集 + 文件落盘 + 数据库同步：需求说明 v2

> 更新：2026-03-09，纳入用户 14:54 反馈

---

## 一、目标

构建一套**不依赖 GUI 常驻**的行情采集体系：

1. 持续采集实时行情（CTP / TTS）
2. 落盘为 CSV 文件（唯一可信数据源）
3. GUI 启动 DataRecorder 时，先把 CSV 同步到数据库
4. 策略继续只从数据库预热（`load_bar(use_database=True)`）
5. 运行中由 GUI DataRecorder 继续增量写数据库
6. 支持 GUI 中录制合约的热更新（事件驱动）

---

## 二、核心设计原则

### 原则 1：CSV 文件 = 唯一可信数据源

数据库（`.vntrader/database.db`）**不可靠**：
- GUI 频繁关闭 / debug / 清库做实验
- 数据库随时可能被清空或重建

因此：
- **Headless Recorder 只写 CSV，不写数据库**
- CSV 文件是 ground truth
- 数据库的内容通过同步操作从 CSV 恢复

### 原则 2：Headless Recorder 和 GUI DataRecorder 可同时运行

两者职责不同，不互斥：

| 组件 | 写 CSV | 写数据库 | 生命周期 |
|------|--------|----------|----------|
| Headless Recorder | ✅ | ❌ | 常驻 |
| GUI DataRecorder | ❌ | ✅ | 跟随 GUI |

- Headless Recorder 只负责 tick → 1m bar → CSV
- GUI DataRecorder 保持原逻辑（tick → bar → 数据库）
- 两者各管各的输出，**没有冲突**

### 原则 3：数据库同步由 GUI DataRecorder 启动触发

- 启动 GUI + 启动 DataRecorder → 先执行 CSV → DB 同步
- 启动 GUI 但没启动 DataRecorder → 不同步（不需要）
- 无头采集器自己永远不碰数据库

### 原则 4：策略不直接读文件

策略层零改动：
- 启动前 `load_bar(..., use_database=True)` 预热
- 实时运行吃 gateway 推送的 tick/bar
- **文件层对策略完全透明**

---

## 三、模块划分

```
┌─────────────────────────────────────────────────┐
│                  CTP / TTS Gateway              │
│              (tick 实时行情推送)                  │
└────────┬──────────────────────┬──────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌──────────────────┐
│ Headless         │    │ GUI              │
│ Recorder         │    │ DataRecorder     │
│                  │    │                  │
│ tick → 1m bar    │    │ tick → bar → DB  │
│ → CSV 落盘       │    │ (原逻辑不变)      │
│                  │    │                  │
│ 常驻运行          │    │ 跟随 GUI          │
└──────┬──────────┘    └────┬─────────────┘
       │                     │
       ▼                     │ 启动时触发
┌─────────────────┐          │
│ CSV 文件         │◄─────────┤
│ (ground truth)  │          │
└──────┬──────────┘    ┌─────▼─────────────┐
       │               │ DB Sync           │
       └──────────────►│ (CSV → database)  │
                       └──────┬────────────┘
                              ▼
                       ┌──────────────┐
                       │ database.db  │
                       │ (可清可重建)  │
                       └──────────────┘
                              ▲
                              │ load_bar()
                       ┌──────┴──────┐
                       │ CTA Strategy │
                       └─────────────┘
```

---

## A. Headless Recorder（无头采集器）

### 职责

- 独立常驻运行（不依赖 GUI 进程）
- 连接 CTP / TTS gateway
- 订阅合约行情
- 接收 tick，通过 BarGenerator 聚合成 1m bar
- 完整 bar 落盘 CSV
- **不写数据库**

### 配置来源

直接读取 `.vntrader/data_recorder_setting.json`：

```json
{
  "tick": {
    "p2605.DCE": { "symbol": "p2605", "exchange": "DCE", "gateway_name": "CTP" }
  },
  "bar": {
    "p2605.DCE": { "symbol": "p2605", "exchange": "DCE", "gateway_name": "CTP" }
  }
}
```

读取 `bar` 部分的合约列表，作为 1m bar 采集目标。
读取 `tick` 部分的合约列表，作为 tick 采集目标。

### 热更新：事件驱动

**不用 mtime 轮询。** 直接实现事件驱动。

#### 实现方案

vnpy `RecorderEngine` 在 `add_bar_recording()` / `remove_bar_recording()` 时会：
1. 修改内存中的 `self.bar_recordings` 字典
2. 调用 `save_setting()` 写回 JSON 文件
3. 发出 `EVENT_RECORDER_UPDATE` 事件（包含最新合约列表）

Headless Recorder 的热更新可以利用两条路径：

**路径 A：共享 EventEngine**（推荐，如果 Headless Recorder 和 GUI 在同一进程）
- 注册 `EVENT_RECORDER_UPDATE` 事件监听
- GUI 增删合约 → RecorderEngine 发事件 → Headless Recorder 回调中更新订阅和 CSV writer

**路径 B：文件 watcher + 配置 diff**（Headless Recorder 独立进程时）
- 使用 `watchdog` 库监听 `data_recorder_setting.json` 的 `IN_MODIFY` 事件
- 检测到变化 → 重新加载 JSON → diff 新旧合约列表 → 增减订阅

```python
# 路径 B 伪代码
from watchdog.observers import Observer
from watchdog.events import FileModifiedEvent, FileSystemEventHandler

class SettingWatcher(FileSystemEventHandler):
    def on_modified(self, event: FileModifiedEvent):
        if event.src_path.endswith("data_recorder_setting.json"):
            new_setting = load_json(self.setting_path)
            added = set(new_setting["bar"]) - set(self.current_bars)
            removed = set(self.current_bars) - set(new_setting["bar"])
            for vt in added:
                self.recorder.subscribe(vt)
            for vt in removed:
                self.recorder.unsubscribe(vt)
            self.current_bars = new_setting["bar"]
```

#### 热更新触发时机

| 时机 | 触发方式 |
|------|----------|
| 采集器启动 | 读取配置 |
| GUI DataRecorder 启动 | save_setting → 文件变化 → watcher 回调 |
| GUI 添加合约 | `add_bar_recording()` → save_setting → watcher |
| GUI 删除合约 | `remove_bar_recording()` → save_setting → watcher |

#### 热更新语义

- **添加合约**：新增订阅 + 创建 CSV writer，从下一笔 tick 开始
- **删除合约**：停止该合约新数据写入，flush 当前缓冲，**已有 CSV 文件保留**

### 参考：vnpy 官方无头录制脚本

`vendor/vnpy/examples/data_recorder/data_recorder.py` 已提供完整范例：
- 创建 `EventEngine` + `MainEngine`
- 添加 CTP Gateway + DataRecorderApp
- 注册 `EVENT_CONTRACT` 自动订阅
- 无 GUI，`input()` 阻塞保持运行

我们的 Headless Recorder 基于此改造：**把 `record_bar()` 从写数据库改为写 CSV。**

---

## B. File Sink（CSV 落盘层）

### 文件路径

```
data/recordings/
  bar_1m/
    p2605.DCE/
      2026-03-09.csv
      2026-03-10.csv
    IF2602.CFFEX/
      2026-03-09.csv
  tick/                 
    p2605.DCE/
      2026-03-09.csv
```

- 按合约 + 日期分文件
- 每天一个新文件（夜盘 21:00 开始算次日）

### 1m Bar CSV 格式

**必须和 Wind 数据口径对齐。**

现有 Wind CSV 格式（`data/analyse/p2509_1min_*.csv`）：
```csv
datetime,open,high,low,close,volume,open_interest,turnover
2025-03-03 09:01:00,8438.0,8456.0,8426.0,8438.0,2129,137493,179730520.0
```

Headless Recorder 写出的 CSV **必须完全一致**：
- 列名：`datetime,open,high,low,close,volume,open_interest,turnover`
- datetime 格式：`YYYY-MM-DD HH:MM:SS`（Asia/Shanghai，无时区后缀）
- 数值格式：浮点数，和 vnpy BarData 字段映射

```python
# 字段映射
CSV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume", "open_interest", "turnover"]

def bar_to_csv_row(bar: BarData) -> dict:
    return {
        "datetime": bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "open": bar.open_price,
        "high": bar.high_price,
        "low": bar.low_price,
        "close": bar.close_price,
        "volume": bar.volume,
        "open_interest": bar.open_interest,
        "turnover": bar.turnover,
    }
```

### 写入时机

- **1m Bar**：只在 bar 完整收盘后写入（`BarGenerator.on_bar` 回调）
- 不写"未完成分钟 bar"
- 写入后立即 flush（`csv_writer.flush()` + `os.fsync()`），防止意外丢失

### 夜盘日期归属

vnpy 的 BarGenerator 产出的 bar.datetime 已包含正确时间：
- `2026-03-09 21:01:00` → 文件归属到 `2026-03-10.csv`（次日交易日）

日期归属逻辑：
```python
def trading_date(dt: datetime) -> str:
    """将行情时间转为交易日期字符串"""
    if dt.hour >= 21:  # 夜盘归属次日
        return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")
```

---

## C. DB Sync（数据库同步器）

### 职责

将 CSV 文件增量同步到 `database.db`，供策略 `load_bar()` 使用。

### 触发时机

**只在 GUI 启动 DataRecorder 时触发**：

```
GUI 启动
  → 用户点击"启动"DataRecorder
    → 先执行 DB Sync（CSV → database.db）
    → 同步完成后，DataRecorder 开始正常运行（实时写 DB）
```

- 如果 GUI 没有启动 DataRecorder → 不触发同步
- Headless Recorder 自己永远不触发同步

### 集成点

在 `RecorderEngine.__init__()` 或 `RecorderEngine.start()` 中注入同步逻辑：

```python
# 在 RecorderEngine 初始化/启动时
from qp.recorder.db_sync import sync_recordings_to_db

class PatchedRecorderEngine(RecorderEngine):
    def start(self) -> None:
        # 同步 CSV → DB（在开始实时录制之前）
        self.write_log("正在同步 CSV 数据到数据库...")
        stats = sync_recordings_to_db(
            recordings_dir=Path("data/recordings/bar_1m"),
            database=self.database,
        )
        self.write_log(f"同步完成: {stats}")
        
        # 然后正常启动
        super().start()
```

### 同步逻辑

```python
def sync_recordings_to_db(recordings_dir: Path, database) -> dict:
    """
    将 CSV 录制文件增量同步到数据库。
    
    Returns:
        {"synced": 150, "skipped": 3200, "errors": 0}
    """
    state = load_sync_state()  # .vntrader/recording_sync_state.json
    stats = {"synced": 0, "skipped": 0, "errors": 0}
    
    for symbol_dir in recordings_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
        
        vt_symbol = symbol_dir.name  # e.g. "p2605.DCE"
        symbol, exchange = parse_vt_symbol(vt_symbol)
        
        for csv_file in sorted(symbol_dir.glob("*.csv")):
            last_dt = state.get(f"{vt_symbol}/{csv_file.name}", None)
            bars = csv_to_bars_incremental(csv_file, symbol, exchange, last_dt)
            
            if bars:
                database.save_bar_data(bars)
                state[f"{vt_symbol}/{csv_file.name}"] = bars[-1].datetime.isoformat()
                stats["synced"] += len(bars)
            else:
                stats["skipped"] += 1
    
    save_sync_state(state)
    return stats
```

### 去重保障

vnpy SQLite 已有 **upsert 机制**：
- `DbBarData` 唯一索引：`(symbol, exchange, interval, datetime)`
- 写入用 `insert_many().on_conflict_replace()`
- **重复导入同一条 bar 不会产生重复记录**

因此即使同步窗口和实时录制有重叠，也不会出问题。

### 同步状态文件

```json
// .vntrader/recording_sync_state.json
{
  "p2605.DCE/2026-03-09.csv": "2026-03-09 14:59:00",
  "p2605.DCE/2026-03-10.csv": "2026-03-10 10:15:00",
  "IF2602.CFFEX/2026-03-09.csv": "2026-03-09 15:00:00"
}
```

记录每个文件已同步到的最后 datetime，下次只导增量。

---

## 四、启动顺序

### 场景 1：纯无头运行（无 GUI）

```
1. Headless Recorder 启动
   → 读取 data_recorder_setting.json
   → 连接 CTP/TTS
   → tick → 1m bar → CSV
   → 常驻运行
```

- 不涉及数据库
- 不涉及策略

### 场景 2：GUI + 策略

```
1. Headless Recorder 已在运行（常驻）
2. 启动 GUI
3. 用户点击启动 DataRecorder
   → DB Sync: CSV → database.db（增量同步）
   → DataRecorder 开始实时写 DB
4. 启动 CTA 策略
   → load_bar(use_database=True) 预热
   → 实时 tick/bar 由 gateway 推送
```

### 场景 3：GUI 关闭后重开

```
1. Headless Recorder 一直在跑（CSV 持续积累）
2. GUI 关闭（DataRecorder 停止，DB 可能被清）
3. 重新打开 GUI + 启动 DataRecorder
   → DB Sync 把关闭期间积累的 CSV 同步到 DB
   → DataRecorder 继续实时写 DB
4. 数据无缝衔接
```

---

## 五、切换时的数据连续性保障

### 问题：DB Sync 和 DataRecorder 同时操作

不会冲突：
- DB Sync 先执行完，DataRecorder 再开始
- 串行，不并发

### 问题：Headless Recorder 正在写 CSV，DB Sync 正在读

**安全**，因为：
- 1m bar 是完整行写入 + flush
- DB Sync 按行读取，最多丢最后一个未完成的换行
- 即使少导最后一条，下次同步会补上

### 问题：夜盘到日盘切换

- 21:00 夜盘开始 → CSV 写入 `次日.csv`
- 次日 09:00 日盘 → 继续写同一个文件
- 无断裂

---

## 六、文件结构

```
src/qp/
  recorder/
    __init__.py
    headless.py           # Headless Recorder 主程序
    csv_sink.py           # CSV 写入器
    db_sync.py            # CSV → DB 同步器
    config_watcher.py     # 配置热更新（watchdog）
    constants.py          # CSV 列定义、路径常量
```

---

## 七、验收标准

### 1. CSV 是 ground truth
- 清空数据库 → 启动 GUI + DataRecorder → DB Sync 从 CSV 完整恢复
- 数据库内容和 CSV 一致

### 2. 配置热更新
- GUI 添加合约 → Headless Recorder 自动开始录制该合约
- GUI 删除合约 → Headless Recorder 停止录制，已有文件保留

### 3. 格式对齐
- Headless Recorder 写出的 CSV 和 `data/analyse/p2509_1min_*.csv` 格式一致
- 能直接走现有 `ingest_vnpy.py` 或新同步器导入

### 4. 不丢数据
- GUI 频繁开关不影响 CSV 连续性
- 清库后重新同步，数据完整

### 5. 同时运行无冲突
- Headless Recorder（写 CSV）+ GUI DataRecorder（写 DB）同时运行正常
- 无文件锁冲突、无 DB 锁冲突

### 6. 可审计
- `data/recordings/bar_1m/<symbol>/` 下文件可直接打开查看
- `.vntrader/recording_sync_state.json` 显示同步进度
- 日志输出同步了多少条、跳过了多少

---

## 八、MVP 范围

### 第一版实现

| 项 | 范围 |
|----|------|
| 采集类型 | 只做 1m bar tick也做 |
| 落盘格式 | CSV，对齐 Wind |
| 同步目标 | `dbbardata` 表 |
| 热更新 | watchdog 文件监听（路径 B） |
| 同步触发 | GUI DataRecorder 启动时 |
| 同步方式 | 增量（基于 state 文件） |

---

## 九、一句话总结

> **Headless Recorder 常驻运行，只写 CSV（ground truth）；GUI DataRecorder 启动时先从 CSV 同步数据库，再正常录制；两者可同时运行互不冲突；策略只读数据库，对文件层无感知。**
