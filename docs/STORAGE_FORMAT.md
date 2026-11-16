# MiniVectorDB 存储与恢复文件格式说明

所有多字节整数与浮点均为 **Little-Endian**。

## 1. 目录结构（`/data`）

```text
/data/
├── checkpoint.meta          # 原子替换的检查点元数据
├── wal/
│   └── wal_<seq>.log        # DDL 与跨文件事务的提交记录（不含行数据）
├── tables/
│   └── t_<tid>_g<gen>.dat   # 表数据文件 = 该表的提交日志（行数据只写这一份）
└── tmp/                     # 原子替换用临时文件（启动时清空）
```

## 2. 表数据文件 `t_<tid>_g<gen>.dat`

连续的帧：

| 偏移 | 字段 | 大小 | 说明 |
| --- | --- | --- | --- |
| 0 | `magic` | 4 B | `b"MVD1"` |
| 4 | `op` | 1 B | `1 Put`、`2 Tombstone`、`3 Commit`、`4 Pending` |
| 5 | `row_id` | 8 B | 行帧为稳定行号；`Pending` 帧为事务号；`Commit` 帧为 0 |
| 13 | `payload_len` | 4 B | |
| 17 | `payload` | `payload_len` B | `Put`：行编码；结束符帧：可选的零填充 |
| 17 + len | `crc32` | 4 B | 覆盖 `magic..payload` |

**组**：一个事务对本表的全部行帧 + 一个结束符帧。

- `Commit`：该组自身即已提交（自动提交语句、单表事务）。
- `Pending(txn)`：该组属于跨文件事务，是否提交取决于 WAL 中有无 `txn` 的提交记录（或是否已被检查点覆盖）。
- 没有结束符、帧残缺或 CRC 不符的尾部：未提交，恢复时截断。
- 组 ≥ 16 KiB 时，结束符帧的 payload 用零填充使组尾落在 4 KiB 边界。

**行编码（`Put` payload）**：`col_count: u16`，随后每列 `tag: u8`：`0 Null`；`1 BigInt` + `i64`；`2 Varchar` + `byte_len: u32` + UTF-8；`3 Vector` + `dim: u32` + `dim × f32`。

## 3. WAL `wal/wal_<seq>.log`

帧：`magic b"WAL2" (4B) + txn_id (8B) + kind (1B) + payload_len (4B) + payload + crc32 (4B)`。

`kind`：`1 Begin`、`2 CreateTable`（表结构）、`3 CreateIndex`（表号、索引名、列名、列序号、维度）、`6 Commit`。

一次追加写入 `Begin + 若干 DDL 帧 + Commit` 并 `fdatasync`；带完整 `Commit` 帧的事务才算提交。

## 4. `checkpoint.meta`

`magic b"CKP2"`、`version = 2`、`wal_seq`、`wal_offset`、`next_txn_id`、`next_table_id`、表数量，随后每张表：表结构（含向量索引元数据）、`generation`、`safe_data_len`（检查点时的数据文件长度）、`next_row_id`；末尾 `crc32` 覆盖全文件。写入流程：`tmp/checkpoint.meta.tmp` → `fsync` → `rename` → 目录 `fsync`。

## 5. 提交与恢复顺序

提交（应答之前）：

1. 单表：`pwrite` 组（以 `Commit` 结束）→ `fdatasync` 表文件 → 更新内存 → 应答。
2. 多表 / 含 DDL：各表 `pwrite` 组（以 `Pending(txn)` 结束）并 `fdatasync` → WAL 追加 `txn` 的提交记录并 `fdatasync`（提交点）→ 更新内存 → 应答。提交点之前出错则把各表文件截回原长度。

恢复：

1. 清空 `tmp/`，读取 `checkpoint.meta`；
2. 从检查点位置扫描 WAL，截断撕裂尾部，收集已提交事务号与 DDL；
3. 打开检查点中的表与 WAL 中已提交的新建表，逐文件扫描：`Commit` 组生效；`Pending(txn)` 组在 `offset < safe_data_len` 或 `txn` 已提交时生效；否则连同其后内容一并截断；
4. 删除无主的数据文件，清理检查点之前的 WAL 段，输出 `{"type":"ready"}`。

恢复过程不写入任何"恢复进度"，中途被强杀后重新执行得到相同结果。
