# MiniVectorDB 简要设计与验证记录（对应第 4.3 节）

## 1. 记录布局与存储设计

- **数据文件**：每张表对应一个 Append-only 二进制堆文件 `/data/tables/t_<tid>_g<gen>.dat`。
- **帧结构（`MVD1`）**：`magic (4B) + op (1B: Put=1 / Tombstone=2) + row_id (8B LE) + payload_len (4B LE) + payload + crc32 (4B LE)`。
- **列编码（显式 Little-Endian）**：依次写入列数 `u16` 与各列类型标签及内容（`BIGINT` 为 `i64` 8 字节、`VARCHAR` 为字节长度 `u32` + UTF-8 字节、`VECTOR` 为维度 `u32` + `f32` LE 数组）。支持选择性解码，评估 `WHERE` 条件时跳过未引用的大向量与长字符串分配。
- **内存轻量索引与有界缓存**：
  - `pk_index: HashMap<i64, RowId>`：`O(1)` 主键查找与唯一性校验。
  - `row_locations: HashMap<RowId, (u64, u32)>`：仅保留稳定 `RowId` 到文件偏移的映射。
  - `BoundedRowCache`（上限 `64 MiB`）：热行全覆盖时内存直接扫描；超过缓存上限时自动淘汰并退化为磁盘流式顺序扫描 + `pread`（`read_exact_at`），避免全量加载导致 2 GiB 容器 OOM。

## 2. 向量索引方案

- **精确检索（`exact`）**：采用 8 路循环展开 + 运行时 AVX2/FMA 加速的平方 L2 距离（`f64` 累加防中间平方溢出），结合先 `WHERE` 过滤后 `BinaryHeap` 最大堆截取 TopK（`O(N log K)` 时间、`O(K)` 空间），距离并列时按主键升序排列，仅最终投影执行 `sqrt()`。
- **近似检索（`ann`）**：自主实现 HNSW 图索引（`M=16, M0=32, ef_construction=120, ef_search=128`，高维大数据量下自动提升搜索 `ef`），结合 AVX2+FMA 32 路展开纯 `f32` 快速距离核（分量 `|x| < 1e15` 时启用）、8 字节紧凑 `Neighbor(id, signed_dist)` 缓存邻居距离与多样性标记、增量反向边启发式剪枝 `add_neighbor_heuristic_incremental` 以及访问代际表（零每查询堆分配）。
  - **过滤 ANN**：保留图节点连通性自由扩展邻居，仅对存活且通过 `WHERE` 谓词的节点入堆；若低选择率导致图检索不足 `K` 条，自动回退至带过滤的精确堆扫描，保证结果有效、无重复且召回率达标。
  - **有界向量溢写**：`VectorStore` 在 `<= 512 MiB` 时常驻连续 `Vec<f32>`；超过阈值自动溢写至 `idx_<tid>_<col>_g<gen>.vec` 并切换为带代数序号的 LRU + `pread`。
  - **流式紧凑快照与免冗余持久化**：`save_snapshot_for_table` 通过 64 KiB `CrcWriter` 流式落盘，在未溢写时采用 `storage_mode = 2`（External Table Store）仅持久化图拓扑与 `u16` 紧凑邻居列表，复用 `.dat` 中的向量数据，将 10,000×128 维索引快照从 `7.01 MiB` 压缩至 `872 KB`；`dirty` 标志与 `shutdown` 缓存预释放机制进一步消除了只读重启容器的落盘开销与关闭阶段的峰值内存冲高。

## 3. 事务处理、提交持久化与崩溃恢复

- **单事务写集**：最多允许一个显式事务（`BEGIN`）。未提交修改暂存在内存 `TxnWriteSet`（按表跟踪新增/修改/删除行及主键增量），显式事务内查询与 `ann` 检索均实时合并读己之修改；单条语句校验或执行失败前不修改写集，确保单语句原子性且保留同事务此前成功语句。
- **提交持久化（WAL 先行）**：无论是 `COMMIT` 还是隐式自动提交，均先将 `Begin` + 操作记录 + `Commit` 帧批量编码写入 `/data/wal/wal_<seq>.log` 并调用 `sync_all()`（`fsync`）落盘，再追加 `.dat` 数据文件并更新索引；`ROLLBACK` 或正常 `shutdown`/EOF 时直接丢弃未提交写集。
- **检查点与幂等 Redo 恢复**：
  1. 启动时清理 `tmp/` 残留，读取经过 CRC32 校验的 `checkpoint.meta`。
  2. 将各表 `.dat` 文件截断回 `checkpoint.meta` 记录的安全长度 `safe_data_len`（清除上次异常终止可能残留的半写尾帧），扫描重构 `pk_index` 与 `row_locations`。
  3. 从检查点 WAL 偏移起顺序扫描 WAL 段，截断尾部不完整或 CRC 失败字节，仅重放包含完整 `COMMIT` 帧的事务。
  4. 恢复中途再次被强杀时，下一次重启仍从同一 `checkpoint.meta` 截断并幂等重放。

## 4. 已实现功能与已知限制

- **已实现**：`CREATE TABLE`（含列级/表级 `BIGINT PRIMARY KEY`、`VARCHAR(n)`、`VECTOR(d)`）、`CREATE VECTOR INDEX`、`INSERT`（单行/多行/指定列）、`SELECT`（`*`/别名/`L2_DISTANCE`）、`WHERE`（`=, !=, <>, <, <=, >, >=, AND, OR, NOT, IS [NOT] NULL, BETWEEN, IN`）、`ORDER BY`（多列与距离别名）、`LIMIT`、`UPDATE`、`DELETE`、`BEGIN`/`COMMIT`/`ROLLBACK`、精确与 HNSW 近似检索、WAL + Checkpoint 崩溃恢复、JSONL STDIO 协议及 Docker 封装。
- **已知限制**：
  - 按赛题单机单客户端定位，不支持多客户端并发、MVCC 或分布式；
  - 不支持 `JOIN`、聚合函数（`COUNT`/`SUM` 等）、子查询、`ALTER TABLE`、`DROP TABLE`、`GROUP BY`/`HAVING`/`OFFSET`，遇到时返回 `unsupported_sql`；
  - 轻量索引 `pk_index` 与 `row_locations` 常驻内存（约数十字节/行），当单表行数达到千万级以上时元数据本身仍会受容器内存上限约束。

## 5. 本地验证记录摘要

- **Rust 测试套件 (`cargo test`)**：5 个测试文件共 14 个用例全部通过（包含 SQL 语义、Unicode/大整数、存储重启与压缩、AVX2 距离等价性、HNSW 召回与磁盘溢写、事务原子性与 WAL/数据文件截断故障注入、STDIO 协议端到端测试）。
- **Docker 公开自测 (`protocol_check/run.py`)**：在 `--network=none --cpus=2 --memory=2g --read-only --cap-drop=ALL` 限制下握手与全部 20 个公开用例 20/20 通过，含 `SIGKILL` 崩溃恢复验证与 `README` 第 7 节空目录全流程验证。
- **性能基准实测**：详见 [性能测试与优化记录](BENCHMARK.md)。
