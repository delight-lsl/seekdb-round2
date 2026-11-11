# MiniVectorDB SQL 语法与通信协议说明

## 1. JSONL STDIO 协议

1. **启动握手**：完成数据目录恢复后向 `stdout` 发送 `{"type":"ready"}\n` 并立即 `flush()`。
2. **请求格式**：
   - `{"id": <json_val>, "op": "execute", "sql": "...", "search_mode": "exact" | "ann"}`
   - `{"id": <json_val>, "op": "shutdown"}`
3. **响应格式**：
   - 查询成功：`{"id": <id>, "ok": true, "columns": [...], "rows": [[...], ...]}`
   - 修改成功：`{"id": <id>, "ok": true, "affected_rows": <usize>}`
   - DDL/事务成功：`{"id": <id>, "ok": true}`
   - 失败响应：`{"id": <id>, "ok": false, "error": {"code": "<error_code>", "message": "<detail>"}}`
   - 关闭响应：`{"id": <id>, "ok": true}`（随后退出码 0 退出，不再向 stdout 输出任何字符）
4. **日志隔离**：所有调试与恢复日志均输出至 `stderr`。

---

## 2. 支持的 SQL 语句

- **DDL**
  - `CREATE TABLE <table_name> (<col_name> <BIGINT [PRIMARY KEY] | VARCHAR(n) | VECTOR(d)> [, ...]);`
  - `CREATE VECTOR INDEX <index_name> ON <table_name>(<vector_col_name>);`
- **DML**
  - `INSERT INTO <table_name> [(<col1>, ...)] VALUES (<v1>, ...) [, (<v1>, ...)];`
  - `SELECT <* | col [AS alias] | L2_DISTANCE(col, '[...]') [AS alias], ...> FROM <table_name> [WHERE <expr>] [ORDER BY <col|alias|expr> [ASC|DESC], ...] [LIMIT <n>];`
  - `UPDATE <table_name> SET <col> = <literal> [, ...] [WHERE <expr>];`
  - `DELETE FROM <table_name> [WHERE <expr>];`
- **TCL**
  - `BEGIN;` / `BEGIN TRANSACTION;`
  - `COMMIT;`
  - `ROLLBACK;`

---

## 3. 语义约束与错误码

| 语义错误码 | 触发场景 |
| --- | --- |
| `unsupported_sql` | `ALTER TABLE`、`DROP TABLE`、`JOIN` 等未支持语法 |
| `syntax_error` | 词法或语法格式错误 |
| `table_exists` | 创建已存在的表 |
| `table_not_found` | 访问不存在的表 |
| `column_not_found` | 引用不存在的列 |
| `index_exists` | 索引已存在或列上已建索引 |
| `duplicate_key` | 主键冲突（含单条批量 INSERT 内部或 UPDATE 修改主键冲突） |
| `dimension_mismatch` | 插入/查询向量维度与表定义不一致 |
| `invalid_vector` | 向量含 `NaN`、`Infinity`、`f32` 溢出或非 `[...]` 格式 |
| `value_too_long` | `VARCHAR(n)` 字符数超过 `n` |
| `type_mismatch` | 类型不匹配（如给 BIGINT 赋值字符串） |
| `transaction_active` | 已处于显式事务中再次执行 `BEGIN` |
| `no_active_transaction` | 未开启事务执行 `COMMIT` / `ROLLBACK` |
| `io_error` / `internal_error` | 磁盘或内部状态异常 |
