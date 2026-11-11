# MiniVectorDB 存储、WAL 与检查点二进制格式说明

所有多字节整数与浮点均采用显式 **Little-Endian (LE)** 编码，严禁直接内存拷贝（transmute / dump）Rust 结构体字节。

---

## 1. 目录结构（`<data_dir>`，容器内默认 `/data`）

```text
/data/
├── checkpoint.meta         # 原子替换的全局检查点元数据（含所有表 Schema、安全截断长度与 WAL 起点）
├── wal/
│   └── wal_00000001.log    # 分段式 Write-Ahead Log
├── tables/
│   ├── t_<tid>_g<gen>.dat  # Append-only 表数据文件（按表 ID 与代数编号）
│   └── idx_<tid>_<col>_g<gen>.hnsw  # 持久化的 HNSW 图索引快照
└── tmp/                    # 临时文件（启动恢复时自动清空）
```

---

## 2. 数据文件格式（`t_<tid>_g<gen>.dat`）

文件由连续的**变长二进制记录帧（Record Frame）**构成：

| 偏移 | 字段 | 大小 | 说明 |
| --- | --- | --- | --- |
| 0 | `magic` | 4 B | 固定魔数 `b"MVD1"` (`0x3144564D`) |
| 4 | `op` | 1 B | `1 = Put`（插入或更新后版本），`2 = Tombstone`（删除标记） |
| 5 | `row_id` | 8 B | `u64` 稳定行标识 |
| 13 | `payload_len` | 4 B | `u32` payload 字节长度（`Tombstone` 时为 `0`） |
| 17 | `payload` | `payload_len` B | 行编码字节（见下文） |
| 17 + `payload_len` | `crc32` | 4 B | 覆盖 `magic..payload` 的 CRC32 校验和 |

### 行编码格式（`Put` Payload）

- `col_count: u16`
- 对每列顺序编码：
  - `tag: u8` (`0 = Null`, `1 = BigInt`, `2 = Varchar`, `3 = Vector`)
  - `BigInt`: `i64` (8 B LE)
  - `Varchar`: `byte_len: u32` + UTF-8 bytes
  - `Vector`: `dim: u32` + `dim` 个 `f32` (`to_le_bytes()`, 4 B each)

> **内存控制**：内存索引仅保留 `pk_index: HashMap<i64, RowId>` 与 `row_locations: HashMap<RowId, (u64, u32)>`（文件偏移与长度）。大字段与超内存数据时通过 `pread` 按需读取与流式扫描，不将全部行载荷常驻堆内存。

---

## 3. WAL 文件格式（`wal/wal_<seq>.log`）

每条日志帧格式如下：

| 字段 | 大小 | 说明 |
| --- | --- | --- |
| `magic` | 4 B | 固定魔数 `b"WAL1"` |
| `txn_id` | 8 B | `u64` 单调递增事务 ID |
| `kind` | 1 B | `1 = Begin`, `2 = CreateTable`, `3 = CreateIndex`, `4 = PutRow`, `5 = DeleteRow`, `6 = Commit` |
| `payload_len` | 4 B | `u32` |
| `payload` | `payload_len` B | 操作编码 |
| `crc32` | 4 B | 覆盖 `magic..payload` 的 CRC32 |

### 持久化与提交协议（Redo-only）

1. 显式事务在内存（或超限溢写）WriteSet 中暂存变更，**未提交前绝不写入 WAL 有效提交序列，也不写入正式数据文件**。
2. `COMMIT` 或自动提交语句时：
   - 将事务所有操作与 `Commit` 帧顺序写入当前 WAL 段；
   - 调用 `file.sync_all()`（`fsync`）确保 WAL 帧（含 `Commit`）可靠落盘；
   - 再将数据变更追加写入对应 `t_<tid>_g<gen>.dat` 并更新内存索引/HNSW 索引；
   - 若写入数据文件途中进程被强杀，重启时会将数据文件截断至上一个 Checkpoint 安全长度并由 WAL Redo 完整重放。

---

## 4. 检查点文件格式（`checkpoint.meta`）

- 写入流程：`checkpoint.meta.tmp` -> `fsync(tmp)` -> `rename(..., checkpoint.meta)` -> `fsync(data_dir)` -> 清理旧 WAL 段与旧代际 `.dat`/`.hnsw` 文件 -> `fsync(data_dir)`。
- 内容：
  - `magic` (`b"CKP1"`), `version` (`u32`)
  - `checkpoint_wal_seq: u64`, `checkpoint_wal_offset: u64`, `next_txn_id: u64`, `next_table_id: u32`
  - 每张表的 `TableSchema`、`generation: u64`、`safe_data_len: u64`、`next_row_id: u64` 及关联 `VectorIndexSchema`
  - 尾部 `crc32: u32` 覆盖全文件校验。

### 崩溃恢复（Redo）流程

1. 清理 `tmp/` 残留临时文件。
2. 加载并校验 `checkpoint.meta`（若不存在则视为全新空库）。
3. 按 `checkpoint.meta` 将每张已记录表的数据文件截断（`set_len(safe_data_len)`）至安全长度；删除不属于检查点的孤儿数据文件；顺序流式扫描安全区间的 `.dat` 文件重建 `pk_index` 与 `row_locations`。
4. 从 `(checkpoint_wal_seq, checkpoint_wal_offset)` 起扫描 WAL 段：
   - 若遇到尾部残缺或 CRC 校验失败帧，立即截断该 WAL 段至上一条完整帧末尾并 `fsync`。
   - 收集所有拥有完整 `Commit` 帧的 `txn_id`（无 `Commit` 的事务一律忽略）。
   - 按原顺序重放所有已提交事务的操作到表元数据与数据文件，并同步 `fsync`。
5. 若重放了 WAL 记录或文件达到阈值，可安全执行一次 Checkpoint；最后加载或重建 HNSW 索引。
6. 恢复全部完成后，向 `stdout` 发送 `{"type":"ready"}`。
