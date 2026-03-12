"""
无头采集 + CSV 落盘 + 数据库同步模块.

- headless: 无头采集器主程序（常驻，只写 CSV）
- csv_sink: CSV 落盘写入器
- db_sync: CSV → database.db 增量同步器
- config_watcher: data_recorder_setting.json 事件驱动热更新
"""
