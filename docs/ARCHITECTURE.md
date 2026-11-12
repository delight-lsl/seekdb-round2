# MiniVectorDB 架构设计文档（阶段一）

## 1. 系统定位与约束

- **运行模式**：单机、单进程、单客户端、stdin/stdout JSONL 通信协议（兼容 `--read-only` 根文件系统，仅 `/data` 可写）。
- **并发与事务**：最多一个显式事务（`BEGIN` / `COMMIT` / `ROLLBACK`），支持语句级原子性；自动提交模式下每条 DDL/DML 具备隐式单语句原子性。
- **内存约束**：严格控制堆内存上限（`<= 2 GiB`，0 swap），数据记录不要求全量常驻内存；支持超过内存限额的向量与记录扫描/检索。
- **持久化目录**：默认 `/data`（本地可通过 `--data-dir` 或 `VECTORDB_DATA_DIR` 覆盖）。

---

## 2. 模块依赖与职责

```text
cli::protocol ──▶ sql::parser & sql::executor
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
catalog::manager   vector::search   transaction::manager
                        │               │
                        ▼               ▼
                  vector::index    recovery::wal & checkpoint
                        │               │
                        └──────▶ storage::engine ◀────── recovery::redo
```

### 各模块核心职责

1. **`common/`**
   - `error.rs`：`DbError` 结构（含语义 `ErrorCode` 与说明消息），涵盖 `duplicate_key`、`dimension_mismatch`、`unsupported_sql`、`syntax_error`、`invalid_vector`、`value_too_long`、`table_exists`、`table_not_found`、`column_not_found`、`index_exists`、`type_mismatch`、`transaction_active`、`no_active_transaction`、`io_error`、`internal_error`。
   - `types.rs`：`DataType`（`BigInt` / `Varchar(usize)` / `Vector(usize)`）、`Value`（`BigInt(i64)` / `Varchar(String)` / `Vector(Vec<f32>)` / `Null`）、`RowId(u64)`、`SearchMode`（`Exact` / `Ann`）。

2. **`cli/`**
   - `protocol.rs`：实现 JSONL 协议：
     1. 启动时完成 WAL 恢复和目录加载后，主动向 `stdout` 输出 `{"type":"ready"}\n` 并立即 `flush()`。
     2. 逐行读取 `stdin` JSON 请求，原样回传 `id` 和布尔 `ok`。
     3. 处理 `op: "execute"` 和 `op: "shutdown"`（回滚未提交事务并清理落盘后返回 `{"id":...,"ok":true}` 并退出码 0）。
     4. 所有诊断日志输出到 `stderr`。

3. **`sql/`**
   - `lexer.rs`：单遍扫描词法分析器，支持 UTF-8 字符串字面量（含 `''` 转义）、大整数原始文本保留（避免 `i64::MIN` 溢出）、负号、浮点、括号、比较与逻辑符号。
   - `ast.rs`：定义 `Statement`（`CreateTable`、`CreateVectorIndex`、`Insert`、`Select`、`Update`、`Delete`、`Begin`、`Commit`、`Rollback`）、`Expr`、`SelectItem`、`OrderByItem`。
   - `parser.rs`：递归下降解析器，不支持的 SQL（如 `ALTER TABLE`）直接返回 `unsupported_sql`。
   - `executor.rs`：
     - 类型与约束校验（`VARCHAR(N)` 字符长度校验、`VECTOR(D)` 维度与 `f32` 有限性检查、主键唯一性检查）。
     - 事务透明读写视图（Overlay）：显式事务中 `SELECT`、`UPDATE`、`DELETE` 均能读取自身未提交修改。
     - 过滤下推与向量 TopK 执行：先评估 `WHERE` 条件再计算 L2 距离，并按 `(distance_sq, primary_key)` 进行确定性排序与并列破除。

4. **`catalog/`**
   - `schema.rs`：`TableSchema`（`table_id`、`name`、`columns`、`pk_col_idx`、`indexes`）、`ColumnSchema`、`VectorIndexSchema`。
   - `manager.rs`：表结构与索引元数据管理，支持在事务中暂存 DDL 修改并在回滚时丢弃。

5. **`storage/`**
   - `codec.rs` / `record.rs`：显式字节序（Little-Endian）二进制编解码 + `CRC32` 校验，绝不直接 dump Rust 结构体内存布局。
   - `file.rs`：Append-only 表数据文件 `tables/t_<tid>_g<gen>.dat`，支持流式顺序读取和基于 `pread` (`FileExt::read_exact_at`) 的随机按偏移读取，无需全量数据常驻内存。
   - `engine.rs`：维护轻量级内存索引：
     - `pk_index: HashMap<i64, RowId>`（O(1) 主键查找与唯一性检查）
     - `row_locations: HashMap<RowId, u64>` / 紧凑偏移数组（仅存储文件偏移，大字段 `VARCHAR` / `VECTOR` 按需从文件/有界 LRU 缓存读取）
     - 数据文件压缩（Compaction）：Checkpoint 时若死记录比例高则通过新代际文件 `g<gen+1>` 原子替换回收空间。

6. **`vector/`**
   - `distance.rs`：L2 欧氏距离与平方 L2 距离计算。向量以 `f32` 存储，使用 `f64` 累加差值平方以防止中间平方溢出，结合 8 路循环展开（及 x86_64 AVX2/FMA 运行时加速）。TopK 比较使用平方 L2 距离，仅在输出投影列时执行一次 `sqrt()`。
   - `search.rs`：基于 `BinaryHeap` 的精确 TopK 检索，复杂度 `O(N log K)`，支持先过滤后距离计算及按主键升序打破并列。
   - `index.rs`：自主实现的 HNSW 近似索引：
     - 参数：`M = 16`, `M0 = 32`, `ef_construction = 120`, `ef_search = 128`（高维场景自适应扩展搜索 `ef`）。
     - 增量反向边启发式裁剪：在 `Neighbor` 中缓存邻居平方距离与 Tier-1/Tier-2 多样性标记，插入反向边时增量校验遮挡关系，避免全量重算两两距离。
     - 有界向量缓存（Bounded LRU Vector Cache，上限 512 MiB）：向量总大小不超过 512 MiB 时全量常驻连续扁平数组以获最高检索吞吐；超过 512 MiB 时自动淘汰并通过 `pread` 按需读取，避免触发 2 GiB 容器 OOM。
     - 过滤 ANN 检索：遍历图邻居保持连通性（仅将满足 `WHERE` 谓词且未删除的节点加入结果候选），若低选择率导致图检索命中不足 `K` 条，自动回退至带谓词过滤的精确扫描。
     - 流式紧凑快照：支持通过 `CrcWriter` 流式持久化快照到 `tables/idx_<tid>_<col>_g<gen>.hnsw`（表级索引采用 `storage_mode = 2` 外部表向量模式 + `u16` 紧凑邻居编码，消除与 `.dat` 的向量重复落盘）与重启恢复。

7. **`transaction/` & `recovery/`**
   - `operation.rs` & `manager.rs`：维护单事务 `WriteSet`（按语句设置 Savepoint：单条语句中途失败可立即截断回滚该语句全部变更，保留同一事务内此前成功语句）。
   - `wal.rs`：分段 WAL（`wal/wal_<seq>.log`），帧格式含 `Magic + TxnId + OpType + Payload + CRC32`，支持 `sync_data()` / `fsync()`。
   - `checkpoint.rs`：写入 `checkpoint.meta.tmp` -> `fsync` -> `rename` 到 `checkpoint.meta` -> `fsync(data_dir)`，原子切换检查点并清理旧 WAL 段。
   - `redo.rs`：启动恢复流程：
     1. 清理残留 `tmp` 文件，加载 `checkpoint.meta`；
     2. 将各表数据文件截断回 `checkpoint.meta` 记录的安全长度，扫描已落盘记录重建轻量索引；
     3. 按序列扫描后续 WAL 段，截断尾部不完整或 CRC 失败帧；
     4. 仅重做存在完整 `COMMIT` 帧的事务；
     5. 重建或增量同步 HNSW 索引。

---

## 3. 关键复杂度分析

| 操作 / 模块 | 时间复杂度 | 空间复杂度（堆内存） | 说明 |
| --- | --- | --- | --- |
| 主键查询 / 唯一性检查 | `O(1)` 均摊 | `O(N)`（约 24B/行） | `HashMap<i64, RowId>` |
| 精确 TopK (`exact`) | `O(N · D + N log K)` | `O(K)` + 有界扫描缓冲 | 先 `WHERE` 过滤，再用大小为 `K` 的最大堆 |
| 近似 TopK (`ann`, HNSW) | `O(ef · M · D · log N)` | 图结构 `O(N · M)` + 向量缓存 `<= 512 MiB` | 超出 512 MiB 自动转为有界 LRU + `pread` |
| 事务提交 (`COMMIT`) | `O(W)`（写集大小） | `O(W)`（超阈值可溢写） | 先写 WAL 并 `fsync`，再更新数据文件与内存索引 |
| 崩溃恢复 (`Redo`) | `O(N_ckpt + W_wal)` | `O(N)`（轻量索引） | 幂等重做，恢复途中崩溃可从头重放 |
