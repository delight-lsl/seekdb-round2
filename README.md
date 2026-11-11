# MiniVectorDB

从零使用 Rust 实现的轻量级单机向量数据库系统，支持自研 SQL 词法/语法解析、Append-only 持久化存储、AVX2/FMA 加速 L2 距离与 `BinaryHeap` 精确 TopK 检索、有界内存 HNSW 近似向量索引、单事务写集与分段 WAL 崩溃恢复，并通过 JSONL `stdin`/`stdout` 协议在 Docker 容器前台运行。

---

## 1. 语言与版本

- **实现语言**：Rust（Edition `2021`，兼容 Ubuntu `24.04` 源内 `1.75.0` 至服务端 `1.93.1`）
- **核心依赖**（均通过 `Cargo.lock` 与 `vendor/` 离线锁定）：`serde`、`serde_json`、`crc32fast`
- **目标平台**：Linux `x86_64`
- **基础镜像**：
  - 标准多阶段构建 `Dockerfile`：`ubuntu:24.04`（编译期：`cargo` + `rustc` 从源码 `--locked --offline` 构建）→ `ubuntu:24.04`（运行期）
  - 离线静态构建 `Dockerfile.musl`：`x86_64-unknown-linux-musl` 静态可执行文件 → `busybox:latest`（供无 Docker Hub 访问的内网环境本地使用）

---

## 2. 镜像构建方式

### 方式 A：标准从源码多阶段构建（推荐用于具备基础镜像拉取环境的构建）

根目录 `Dockerfile` 从源码完整编译并打包至 `ubuntu:24.04`：

```bash
docker build -t minivectordb:latest .
```

### 方式 B：使用构建脚本（自动适配本地/离线环境并导出镜像包）

```bash
chmod +x scripts/build_docker.sh scripts/run_docker.sh
./scripts/build_docker.sh minivectordb:latest minivectordb.tar
```

若宿主机 Docker 守护进程无法访问外部镜像仓库，脚本会自动调用本机 `cargo 1.93.1` 编译静态链接的 `x86_64-unknown-linux-musl` 二进制，并通过 `Dockerfile.musl` 打包生成 `minivectordb:latest` 并导出 `minivectordb.tar`。

---

## 3. 启动入口与通信协议

- **镜像入口**：
  - 工作目录：`WORKDIR /app`
  - 启动命令：`ENTRYPOINT ["/app/minivectordb"]`、`CMD ["--data-dir", "/data"]`
  - 支持环境变量 `VECTORDB_DATA_DIR` 或命令行 `--data-dir <path>` 覆盖数据目录（默认 `/data`）。
- **前台运行方式**（不使用 `-d`、不使用 `-t`、不暴露网络端口）：

```bash
docker run --rm -i \
  -v minivectordb-data:/data \
  --cpus=2 \
  --memory=2g \
  minivectordb:latest
```

- **JSONL STDIO 协议**：
  1. 启动后先对 `/data` 执行崩溃恢复与检查点加载，随后主动向 `stdout` 输出 `{"type":"ready"}\n` 并立即 `flush()`。
  2. 从 `stdin` 逐行读取 JSON 请求，逐行返回包含相同 `id` 与布尔值 `ok` 的 JSON 响应并 `flush()`。
  3. 收到 `{"id":...,"op":"shutdown"}` 时，回滚未提交事务、刷新检查点与索引快照，返回 `{"id":...,"ok":true}\n` 后以退出码 `0` 退出，不再向 `stdout` 输出任何内容。
  4. 所有启动与诊断日志均写入 `stderr`，绝不污染 `stdout`。

---

## 4. `/data` 数据目录

数据库持久化文件全部写入 `/data`（在 `--read-only` 只读根文件系统容器下仅 `/data` 可写）：

```text
/data/
├── checkpoint.meta                  # 原子替换的检查点元数据（表结构、安全截断长度、WAL 起点、CRC32）
├── wal/
│   └── wal_00000001.log             # 分段式 Write-Ahead Log（CRC32 校验）
├── tables/
│   ├── t_<tid>_g<gen>.dat           # Append-only 表数据文件（Put / Tombstone 帧）
│   ├── idx_<tid>_<col>_g<gen>.hnsw  # 持久化的 HNSW 图索引快照
│   └── idx_<tid>_<col>_g<gen>.vec   # 超出 512 MiB 内存限额时的磁盘向量溢写文件
└── tmp/                             # 临时原子替换目录（启动时自动清理）
```

重启或崩溃恢复时复用原 `/data` 目录，不清空既有数据；已提交事务通过 WAL Redo 恢复，未提交或无有效 `COMMIT` 帧的事务在恢复及截断尾部残缺帧时被彻底丢弃。

---

## 5. SQL 支持范围

- **数据类型**：
  - `BIGINT`：64 位有符号整数（精确保持 64 位精度，支持 `PRIMARY KEY`）
  - `VARCHAR(n)`：UTF-8 变长字符串（按字符数 `<= n` 校验）
  - `VECTOR(d)`：固定维度 `d` 的 `float32` 向量（支持 `'[x1, x2, ...]'` 字符串或 `[x1, x2, ...]` 字面量）
- **支持语句**：
  - DDL：`CREATE TABLE t (col type [PRIMARY KEY] [NOT NULL], ... [, PRIMARY KEY (col)])`、`CREATE VECTOR INDEX idx ON t(vec_col)`
  - DML：`INSERT INTO t [(cols)] VALUES (...), (...)`、`SELECT ... FROM t [WHERE ...] [ORDER BY ... [ASC|DESC]] [LIMIT k]`、`UPDATE t SET ... [WHERE ...]`、`DELETE FROM t [WHERE ...]`
  - TCL：`BEGIN [TRANSACTION]`、`COMMIT [TRANSACTION]`、`ROLLBACK [TRANSACTION]`
- **查询能力**：
  - `WHERE` 支持 `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, `NOT`, `IS [NOT] NULL`, `[NOT] BETWEEN`, `[NOT] IN (...)` 与括号组合
  - `L2_DISTANCE(vec_col, query_vec)` 返回欧氏距离（内部存储为 `f32`，`f64` 累加防平方溢出；`ORDER BY` 使用平方距离排序且支持别名引用，并列按主键升序排列）
  - `search_mode = "exact"` 使用 `BinaryHeap` 精确 TopK（先过滤后截取）；`search_mode = "ann"` 在存在向量索引时使用 HNSW 图检索（支持过滤与回退）
- **明确不支持并报错（`unsupported_sql`）**：`ALTER TABLE`、`DROP`、`TRUNCATE`、`JOIN`、`GROUP BY`、`HAVING`、`DISTINCT`、`OFFSET`、`CREATE INDEX`（非 `VECTOR`）等。

---

## 6. 错误处理

- **返回格式**：`{"id": <id>, "ok": false, "error": {"code": "<error_code>", "message": "<detail>"}}`。
- **关键错误码**：`duplicate_key`、`dimension_mismatch`、`invalid_vector`（含 `NaN`/`Infinity`/`f32` 溢出）、`value_too_long`、`type_mismatch`、`unsupported_sql`、`syntax_error`、`table_exists`、`table_not_found`、`column_not_found`、`index_exists`、`transaction_active`（嵌套 `BEGIN`）、`no_active_transaction`。
- **原子性与会话连续性**：
  - 单条语句失败（如批量 `INSERT` 第 2 行主键冲突或维度不匹配）保证整条语句零部分生效。
  - 显式事务内某条语句失败后，此前同事务已成功的修改保留，事务仍可继续或执行 `COMMIT`/`ROLLBACK`。
  - 遇到预期 SQL 错误后进程不退出，继续从 `stdin` 接收后续请求。

---

## 7. 从空数据目录开始的完整操作示例

下面演示从全新空数据卷启动容器，依次完成**建表建索引、写入、普通查询、向量检索（exact/ann）、提交、回滚、关闭与重启恢复**：

```bash
# 1. 创建全新持久化目录并第一次启动容器（第 1 轮会话）
mkdir -p ./demo_data && chmod 777 ./demo_data

docker run --rm -i \
  -v "$(pwd)/demo_data:/data" \
  --cpus=2 --memory=2g \
  minivectordb:latest <<'JSONL'
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), content VARCHAR(32), embedding VECTOR(2));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"CREATE VECTOR INDEX docs_vec ON docs(embedding);","search_mode":"exact"}
{"id":3,"op":"execute","sql":"INSERT INTO docs VALUES (1, 'keep', '初始甲', '[0, 0]'), (2, 'skip', '初始乙', '[10, 0]'), (3, 'keep', '初始丙', '[3, 4]');","search_mode":"exact"}
{"id":4,"op":"execute","sql":"SELECT id, content FROM docs WHERE category = 'keep' ORDER BY id;","search_mode":"exact"}
{"id":5,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs WHERE category = 'keep' ORDER BY distance LIMIT 2;","search_mode":"exact"}
{"id":6,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 2;","search_mode":"ann"}
{"id":7,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":8,"op":"execute","sql":"UPDATE docs SET content = '已提交修改', embedding = '[0, 2]' WHERE id = 1;","search_mode":"exact"}
{"id":9,"op":"execute","sql":"COMMIT;","search_mode":"exact"}
{"id":10,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":11,"op":"execute","sql":"DELETE FROM docs WHERE id = 3;","search_mode":"exact"}
{"id":12,"op":"execute","sql":"ROLLBACK;","search_mode":"exact"}
{"id":13,"op":"shutdown"}
JSONL
```

预期 `stdout` 逐行返回：
- `{"type":"ready"}`（启动握手）
- `{"id":1,"ok":true}`、`{"id":2,"ok":true}`（建表与建索引成功）
- `{"id":3,"ok":true,"affected_rows":3}`（写入 3 行）
- `{"id":4,"ok":true,"columns":["id","content"],"rows":[[1,"初始甲"],[3,"初始丙"]]}`（普通查询）
- `{"id":5,"ok":true,"columns":["id","distance"],"rows":[[1,0.0],[3,5.0]]}`（`exact` 过滤 + 向量 TopK）
- `{"id":6,"ok":true,"columns":["id","distance"],"rows":[[1,0.0],[3,5.0]]}`（`ann` HNSW 检索）
- `{"id":7,"ok":true}`、`{"id":8,"ok":true,"affected_rows":1}`、`{"id":9,"ok":true}`（提交事务生效）
- `{"id":10,"ok":true}`、`{"id":11,"ok":true,"affected_rows":1}`、`{"id":12,"ok":true}`（回滚事务撤销删除）
- `{"id":13,"ok":true}`（正常关闭退出）

```bash
# 2. 使用同一数据目录重启容器（第 2 轮会话），验证持久化与回滚结果
docker run --rm -i \
  -v "$(pwd)/demo_data:/data" \
  --cpus=2 --memory=2g \
  minivectordb:latest <<'JSONL'
{"id":101,"op":"execute","sql":"SELECT id, content, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 5;","search_mode":"ann"}
{"id":102,"op":"shutdown"}
JSONL
```

重启后输出：
- `{"type":"ready"}`
- `{"id":101,"ok":true,"columns":["id","content","distance"],"rows":[[1,"已提交修改",2.0],[3,"初始丙",5.0],[2,"初始乙",10.0]]}`（已提交更新保留，已回滚的 `id=3` 仍存在）
- `{"id":102,"ok":true}`

---

## 8. 本地测试、基准与更多文档

```bash
# 运行全部单元、集成、协议与故障注入测试
cargo test

# 运行 Release 性能基准
cargo run --release --example bench
```

- [总体架构与复杂度分析](docs/ARCHITECTURE.md)
- [简要设计与验证记录（第 4.3 节）](docs/DESIGN_AND_VERIFICATION.md)
- [存储与 WAL / Checkpoint 二进制格式](docs/STORAGE_FORMAT.md)
- [SQL 语法与 JSONL 通信协议](docs/SQL_SPEC.md)
- [性能测试与优化记录](docs/BENCHMARK.md)
