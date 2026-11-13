use crate::catalog::manager::Catalog;
use crate::catalog::schema::TableSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{
    parse_vector_str, validate_f64_slice_as_f32, Cell, DataType, ExecResult, Row, RowAccess, RowId,
    SearchMode, TableId, Value,
};
use crate::recovery::checkpoint::perform_checkpoint;
use crate::recovery::redo::recover_database;
use crate::recovery::wal::WalManager;
use crate::sql::ast::{
    BinaryOp, Expr, LiteralValue, OrderByItem, OrderByTarget, QueryVector, SelectItem, Statement,
};
use crate::sql::parser::Parser;
use crate::storage::engine::{mem_budget_bytes, pk_of_row, GroupEnd, StorageEngine};
use crate::storage::mem::MemTable;
use crate::transaction::manager::TransactionManager;
use crate::transaction::operation::WalOp;
use crate::vector::distance::{l2_distance, l2_distance_squared};
use crate::vector::search::{TopKCandidate, TopKHeap};
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Bound expression validated against a `TableSchema`.
#[derive(Debug, Clone)]
pub enum BoundExpr {
    Column(usize),
    Literal(Value),
    FloatLiteral(f64),
    L2Distance {
        col_idx: usize,
        query: Vec<f32>,
    },
    Not(Box<BoundExpr>),
    IsNull {
        expr: Box<BoundExpr>,
        negated: bool,
    },
    Binary {
        left: Box<BoundExpr>,
        op: BinaryOp,
        right: Box<BoundExpr>,
    },
}

#[derive(Debug, Clone)]
pub enum BoundOrderKey {
    Column(usize),
    L2Distance { col_idx: usize, query: Vec<f32> },
}

#[derive(Debug, Clone)]
pub enum SortVal {
    Null,
    Int(i64),
    Str(String),
    Float(f64),
}

impl SortVal {
    pub fn cmp_val(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Null, Self::Null) => Ordering::Equal,
            (Self::Null, _) => Ordering::Greater,
            (_, Self::Null) => Ordering::Less,
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => a.total_cmp(b),
            _ => Ordering::Equal,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BoundSelectItem {
    Column {
        col_idx: usize,
        output_name: String,
    },
    L2Distance {
        col_idx: usize,
        query: Vec<f32>,
        output_name: String,
    },
}

type PutList = Vec<(RowId, Option<i64>, Row)>;
type DeleteList = Vec<(RowId, Option<i64>)>;

/// Complete MiniVectorDB database engine combining SQL execution, storage,
/// exact vector search, single-writer transactions and crash recovery.
pub struct Database {
    pub data_dir: PathBuf,
    pub storage: StorageEngine,
    pub wal: WalManager,
    pub txn_mgr: TransactionManager,
}

impl Database {
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let (storage, wal) = recover_database(&data_dir)?;
        Ok(Self {
            data_dir,
            storage,
            wal,
            txn_mgr: TransactionManager::new(),
        })
    }

    pub fn execute_sql(&mut self, sql: &str, search_mode: SearchMode) -> Result<ExecResult> {
        let stmt = Parser::parse_sql(sql)?;
        match stmt {
            Statement::Begin => {
                self.txn_mgr.begin()?;
                Ok(ExecResult::Ok)
            }
            Statement::Commit => self.exec_commit(),
            Statement::Rollback => {
                self.txn_mgr.rollback()?;
                Ok(ExecResult::Ok)
            }
            Statement::CreateTable { name, columns } => self.exec_create_table(&name, columns),
            Statement::CreateVectorIndex {
                index_name,
                table_name,
                column_name,
            } => self.exec_create_vector_index(&index_name, &table_name, &column_name),
            Statement::Insert {
                table_name,
                columns,
                rows,
            } => self.exec_insert(&table_name, columns.as_deref(), rows),
            Statement::Select {
                items,
                table_name,
                where_clause,
                order_by,
                limit,
            } => self.exec_select(
                &table_name,
                &items,
                where_clause.as_ref(),
                &order_by,
                limit,
                search_mode,
            ),
            Statement::Update {
                table_name,
                assignments,
                where_clause,
            } => self.exec_update(&table_name, &assignments, where_clause.as_ref()),
            Statement::Delete {
                table_name,
                where_clause,
            } => self.exec_delete(&table_name, where_clause.as_ref()),
        }
    }

    /// Normal shutdown: the open transaction (if any) is rolled back. Every acknowledged
    /// commit is already durable, so nothing has to be flushed here; a checkpoint is only
    /// taken when it actually reclaims space.
    pub fn shutdown(&mut self) -> Result<()> {
        self.txn_mgr.discard_if_active();
        if crate::storage::file::sync_policy() != crate::storage::file::SyncPolicy::Full {
            for dt in self.storage.tables.values() {
                dt.file.sync()?;
            }
        }
        let any_table_needs_compact = self.storage.tables.values().any(|dt| dt.needs_compaction());
        if any_table_needs_compact || self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, false)?;
        }
        Ok(())
    }

    fn after_write(&mut self, tid: Option<TableId>) -> Result<()> {
        let compact = match tid {
            Some(t) => self.storage.tables.get(&t).is_some_and(|dt| dt.needs_compaction()),
            None => self.storage.tables.values().any(|dt| dt.needs_compaction()),
        };
        if compact || self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, false)?;
        }
        if self.storage.mem_bytes() > mem_budget_bytes() {
            self.storage.enforce_memory_budget()?;
        }
        Ok(())
    }

    fn effective_catalog(&self) -> Catalog {
        let mut cat = self.storage.catalog.clone();
        if let Some(txn) = &self.txn_mgr.active {
            for schema in &txn.created_tables {
                let _ = cat.insert_table(schema.clone());
            }
            for (tid, idx) in &txn.created_indexes {
                let _ = cat.add_vector_index(*tid, idx.clone());
            }
        }
        cat
    }

    fn resolve_table_schema(&self, table_name: &str) -> Result<TableSchema> {
        if let Some(txn) = &self.txn_mgr.active {
            for t in &txn.created_tables {
                if t.name.eq_ignore_ascii_case(table_name) {
                    let mut schema = t.clone();
                    for (tid, idx) in &txn.created_indexes {
                        if *tid == schema.table_id {
                            schema.indexes.push(idx.clone());
                        }
                    }
                    return Ok(schema);
                }
            }
            let mut schema = self.storage.catalog.get_table(table_name)?.clone();
            for (tid, idx) in &txn.created_indexes {
                if *tid == schema.table_id {
                    schema.indexes.push(idx.clone());
                }
            }
            return Ok(schema);
        }
        Ok(self.storage.catalog.get_table(table_name)?.clone())
    }

    fn exec_create_table(
        &mut self,
        name: &str,
        columns: Vec<crate::catalog::schema::ColumnSchema>,
    ) -> Result<ExecResult> {
        let cat = self.effective_catalog();
        let schema = cat.validate_new_table(name, columns)?;
        if let Some(txn) = &mut self.txn_mgr.active {
            txn.created_tables.push(schema);
            return Ok(ExecResult::Ok);
        }
        self.wal
            .append_committed_txn(&[WalOp::CreateTable(schema.clone())])?;
        self.storage.create_table(schema)?;
        self.after_write(None)?;
        Ok(ExecResult::Ok)
    }

    /// The vector index is a schema-level object: searches are always answered by the
    /// exact SIMD scan over the in-memory vector column, which stays consistent with the
    /// table (and the open transaction) by construction and has recall 1.0 in every mode.
    fn exec_create_vector_index(
        &mut self,
        index_name: &str,
        table_name: &str,
        column_name: &str,
    ) -> Result<ExecResult> {
        let cat = self.effective_catalog();
        let (tid, idx_schema) =
            cat.validate_new_vector_index(index_name, table_name, column_name)?;
        if let Some(txn) = &mut self.txn_mgr.active {
            txn.created_indexes.push((tid, idx_schema));
            return Ok(ExecResult::Ok);
        }
        self.wal.append_committed_txn(&[WalOp::CreateIndex {
            table_id: tid,
            index: idx_schema.clone(),
        }])?;
        self.storage
            .catalog
            .add_vector_index(tid, idx_schema.clone())?;
        self.storage.get_table_mut(tid)?.schema.indexes.push(idx_schema);
        self.after_write(None)?;
        Ok(ExecResult::Ok)
    }

    /// Durably commit one group of changes to a single table, then publish it.
    fn commit_table_group(&mut self, tid: TableId, puts: PutList, deletes: &DeleteList) -> Result<()> {
        let dt = self.storage.get_table_mut(tid)?;
        let locs = dt.append_group(&puts, deletes, GroupEnd::Commit)?;
        dt.apply_group(puts, deletes, locs)?;
        self.after_write(Some(tid))
    }

    fn exec_commit(&mut self) -> Result<ExecResult> {
        let txn = self.txn_mgr.take_for_commit()?;
        let has_ddl = !txn.created_tables.is_empty() || !txn.created_indexes.is_empty();
        let mut writes: Vec<(TableId, PutList, DeleteList)> = Vec::new();
        for (tid, tws) in txn.table_writes {
            if tws.puts.is_empty() && tws.deletes.is_empty() {
                continue;
            }
            let puts: PutList = tws
                .puts
                .into_iter()
                .map(|(rid, (old_pk, row))| (rid, old_pk, row))
                .collect();
            let deletes: DeleteList = tws.deletes.into_iter().collect();
            writes.push((tid, puts, deletes));
        }
        if !has_ddl && writes.is_empty() {
            return Ok(ExecResult::Ok);
        }
        if !has_ddl && writes.len() == 1 {
            // Single-table transaction: its group is self-committing (one fdatasync).
            let (tid, puts, deletes) = writes.pop().unwrap();
            self.commit_table_group(tid, puts, &deletes)?;
            return Ok(ExecResult::Ok);
        }

        // Multi-file transaction: stage every table group as pending, then make the
        // whole transaction durable with a single commit record in the WAL.
        let txn_id = self.wal.alloc_txn_id();
        let mut created: Vec<TableId> = Vec::new();
        let mut undo: Vec<(TableId, u64, u64)> = Vec::new();
        let mut staged: Vec<(TableId, PutList, DeleteList, Vec<(u64, u32)>)> = Vec::new();
        let mut ops: Vec<WalOp> = Vec::new();
        for schema in &txn.created_tables {
            ops.push(WalOp::CreateTable(schema.clone()));
        }
        for (tid, idx) in &txn.created_indexes {
            ops.push(WalOp::CreateIndex {
                table_id: *tid,
                index: idx.clone(),
            });
        }

        let mut res: Result<()> = Ok(());
        for schema in &txn.created_tables {
            match self.storage.create_table(schema.clone()) {
                Ok(()) => created.push(schema.table_id),
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        if res.is_ok() {
            for (tid, puts, deletes) in writes {
                let step = (|| -> Result<Vec<(u64, u32)>> {
                    let dt = self.storage.get_table_mut(tid)?;
                    undo.push((tid, dt.file.len(), dt.total_frames));
                    dt.append_group(&puts, &deletes, GroupEnd::Pending(txn_id))
                })();
                match step {
                    Ok(locs) => staged.push((tid, puts, deletes, locs)),
                    Err(e) => {
                        res = Err(e);
                        break;
                    }
                }
            }
        }
        if res.is_ok() {
            res = self.wal.append_commit(txn_id, &ops);
        }
        if let Err(e) = res {
            // Not committed: remove every trace so later commits are not shadowed.
            for (tid, len, frames) in undo {
                if !created.contains(&tid) {
                    if let Ok(dt) = self.storage.get_table_mut(tid) {
                        let _ = dt.rollback_file_to(len, frames);
                    }
                }
            }
            for tid in created {
                self.storage.drop_uncommitted_table(tid);
            }
            return Err(e);
        }

        for (tid, puts, deletes, locs) in staged {
            self.storage.get_table_mut(tid)?.apply_group(puts, &deletes, locs)?;
        }
        for (tid, idx) in txn.created_indexes {
            self.storage.catalog.add_vector_index(tid, idx.clone())?;
            self.storage.get_table_mut(tid)?.schema.indexes.push(idx);
        }
        self.after_write(None)?;
        Ok(ExecResult::Ok)
    }

    fn pk_exists(&self, tid: TableId, pk: i64) -> bool {
        if let Some(txn) = &self.txn_mgr.active {
            if let Some(tws) = txn.table_writes.get(&tid) {
                if tws.pk_adds.contains_key(&pk) {
                    return true;
                }
                if tws.pk_removes.contains(&pk) {
                    return false;
                }
            }
        }
        self.storage
            .tables
            .get(&tid)
            .is_some_and(|dt| dt.pk_index.contains_key(&pk))
    }

    fn next_row_id_for(&self, tid: TableId) -> RowId {
        let base = self.storage.tables.get(&tid).map_or(1, |dt| dt.next_row_id);
        if let Some(txn) = &self.txn_mgr.active {
            if let Some(tws) = txn.table_writes.get(&tid) {
                if let Some(next_id) = tws.next_row_id {
                    return next_id.max(base);
                }
            }
        }
        base
    }

    fn exec_insert(
        &mut self,
        table_name: &str,
        col_names: Option<&[String]>,
        raw_rows: Vec<Vec<LiteralValue>>,
    ) -> Result<ExecResult> {
        let schema = self.resolve_table_schema(table_name)?;
        let col_mapping = resolve_insert_columns(&schema, col_names)?;

        let mut prepared_rows = Vec::with_capacity(raw_rows.len());
        let mut batch_pks = HashSet::with_capacity(raw_rows.len());

        for raw_vals in raw_rows {
            if raw_vals.len() != col_mapping.len() {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    format!(
                        "column count mismatch: expected {}, got {}",
                        col_mapping.len(),
                        raw_vals.len()
                    ),
                ));
            }
            let mut row_vals = vec![Value::Null; schema.columns.len()];
            for (lit, &col_idx) in raw_vals.into_iter().zip(col_mapping.iter()) {
                let col_schema = &schema.columns[col_idx];
                row_vals[col_idx] =
                    coerce_literal_owned(lit, &col_schema.data_type, col_schema.is_primary_key)?;
            }
            if let Some(pk_idx) = schema.pk_col_idx {
                match &row_vals[pk_idx] {
                    Value::BigInt(pk) => {
                        if self.pk_exists(schema.table_id, *pk) || !batch_pks.insert(*pk) {
                            return Err(DbError::new(
                                ErrorCode::DuplicateKey,
                                format!("duplicate primary key: {}", pk),
                            ));
                        }
                    }
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "primary key cannot be NULL",
                        ));
                    }
                }
            }
            prepared_rows.push(Row { values: row_vals });
        }

        let count = prepared_rows.len();
        let mut next_rid = self.next_row_id_for(schema.table_id);

        if let Some(txn) = &mut self.txn_mgr.active {
            let tws = txn.table_writes.entry(schema.table_id).or_default();
            for row in prepared_rows {
                let rid = next_rid;
                next_rid += 1;
                if let Some(pk) = pk_of_row(&schema, &row) {
                    tws.pk_removes.remove(&pk);
                    tws.pk_adds.insert(pk, rid);
                }
                tws.puts.insert(rid, (None, row));
            }
            tws.next_row_id = Some(next_rid);
            return Ok(ExecResult::AffectedRows(count));
        }

        let mut puts: PutList = Vec::with_capacity(count);
        for row in prepared_rows {
            puts.push((next_rid, None, row));
            next_rid += 1;
        }
        self.commit_table_group(schema.table_id, puts, &Vec::new())?;
        Ok(ExecResult::AffectedRows(count))
    }

    fn collect_matching_rows_limited(
        &self,
        schema: &TableSchema,
        where_expr: Option<&BoundExpr>,
        needed_cols: &[bool],
        limit: Option<usize>,
    ) -> Result<Vec<(RowId, Row)>> {
        let max_rows = limit.unwrap_or(usize::MAX);
        if max_rows == 0 {
            return Ok(Vec::new());
        }
        let mut where_cols = vec![false; schema.columns.len()];
        if let Some(expr) = where_expr {
            collect_expr_cols(expr, &mut where_cols);
        }
        let tid = schema.table_id;
        let tws_opt = self
            .txn_mgr
            .active
            .as_ref()
            .and_then(|t| t.table_writes.get(&tid));

        let mut results = Vec::new();
        if let Some(dt) = self.storage.tables.get(&tid) {
            let pk_probe = where_expr.and_then(|e| pk_eq_literal(schema, e));
            if let (Some(pk), Some(expr)) = (pk_probe, where_expr) {
                // Point lookup through the primary-key index.
                if let Some(&rid) = dt.pk_index.get(&pk) {
                    let hidden = tws_opt
                        .is_some_and(|t| t.deletes.contains_key(&rid) || t.puts.contains_key(&rid));
                    if !hidden {
                        let got = dt.with_row(rid, &where_cols, |row| {
                            if eval_where(expr, &row)? {
                                Ok(Some(row.materialize(needed_cols)?))
                            } else {
                                Ok(None)
                            }
                        })?;
                        if let Some(Some(row)) = got {
                            results.push((rid, row));
                        }
                    }
                }
            } else {
                dt.scan_live(&where_cols, |rid, row| {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            return Ok(true);
                        }
                    }
                    let ok = match where_expr {
                        Some(expr) => eval_where(expr, &row)?,
                        None => true,
                    };
                    if ok {
                        results.push((rid, row.materialize(needed_cols)?));
                        if results.len() >= max_rows {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                })?;
            }
        }

        if results.len() < max_rows {
            if let Some(tws) = tws_opt {
                for (&rid, (_, row)) in &tws.puts {
                    let ok = match where_expr {
                        Some(expr) => eval_where(expr, row)?,
                        None => true,
                    };
                    if ok {
                        results.push((rid, row.clone()));
                        if results.len() >= max_rows {
                            break;
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    fn collect_matching_rows(
        &self,
        schema: &TableSchema,
        where_expr: Option<&BoundExpr>,
        needed_cols: &[bool],
    ) -> Result<Vec<(RowId, Row)>> {
        self.collect_matching_rows_limited(schema, where_expr, needed_cols, None)
    }

    fn exec_update(
        &mut self,
        table_name: &str,
        assignments: &[(String, LiteralValue)],
        where_clause: Option<&Expr>,
    ) -> Result<ExecResult> {
        let schema = self.resolve_table_schema(table_name)?;
        if assignments.is_empty() {
            return Err(DbError::new(
                ErrorCode::SyntaxError,
                "UPDATE SET requires at least one assignment",
            ));
        }

        let mut bound_assigns = Vec::with_capacity(assignments.len());
        let mut seen_assign_cols = HashSet::new();
        for (col_name, lit) in assignments {
            let (col_idx, col_schema) = schema.find_column(col_name).ok_or_else(|| {
                DbError::new(
                    ErrorCode::ColumnNotFound,
                    format!("column '{}' not found in table '{}'", col_name, table_name),
                )
            })?;
            if !seen_assign_cols.insert(col_idx) {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    format!("duplicate column '{}' in SET clause", col_name),
                ));
            }
            let val = coerce_literal(lit, &col_schema.data_type, col_schema.is_primary_key)?;
            bound_assigns.push((col_idx, val));
        }

        let bound_where = match where_clause {
            Some(expr) => Some(bind_where_expr(&schema, expr)?),
            None => None,
        };

        let all_cols = vec![true; schema.columns.len()];
        let candidates = self.collect_matching_rows(&schema, bound_where.as_ref(), &all_cols)?;

        let mut updates: PutList = Vec::with_capacity(candidates.len());
        let mut old_pks_set = HashSet::new();
        let mut new_pks_set = HashSet::new();

        for (rid, row) in candidates {
            let old_pk = pk_of_row(&schema, &row);
            if let Some(opk) = old_pk {
                old_pks_set.insert(opk);
            }
            let mut new_row = row;
            for (col_idx, val) in &bound_assigns {
                new_row.values[*col_idx] = val.clone();
            }
            updates.push((rid, old_pk, new_row));
        }

        if let Some(pk_idx) = schema.pk_col_idx {
            for (_, _, new_row) in &updates {
                let Value::BigInt(npk) = new_row.values[pk_idx] else {
                    return Err(DbError::new(
                        ErrorCode::TypeMismatch,
                        "primary key cannot be NULL",
                    ));
                };
                if !new_pks_set.insert(npk) {
                    return Err(DbError::new(
                        ErrorCode::DuplicateKey,
                        format!("duplicate primary key in update: {}", npk),
                    ));
                }
                if !old_pks_set.contains(&npk) && self.pk_exists(schema.table_id, npk) {
                    return Err(DbError::new(
                        ErrorCode::DuplicateKey,
                        format!("duplicate primary key: {}", npk),
                    ));
                }
            }
        }

        let affected = updates.len();
        if affected == 0 {
            return Ok(ExecResult::AffectedRows(0));
        }

        if let Some(txn) = &mut self.txn_mgr.active {
            let tws = txn.table_writes.entry(schema.table_id).or_default();
            for (rid, old_pk, new_row) in updates {
                let orig_committed_pk = if let Some((prev_orig_pk, _)) = tws.puts.get(&rid) {
                    *prev_orig_pk
                } else {
                    old_pk
                };
                if let Some(opk) = old_pk {
                    tws.pk_adds.remove(&opk);
                    if orig_committed_pk == Some(opk) {
                        tws.pk_removes.insert(opk);
                    }
                }
                if let Some(npk) = pk_of_row(&schema, &new_row) {
                    tws.pk_removes.remove(&npk);
                    tws.pk_adds.insert(npk, rid);
                }
                tws.puts.insert(rid, (orig_committed_pk, new_row));
            }
            return Ok(ExecResult::AffectedRows(affected));
        }

        self.commit_table_group(schema.table_id, updates, &Vec::new())?;
        Ok(ExecResult::AffectedRows(affected))
    }

    fn exec_delete(
        &mut self,
        table_name: &str,
        where_clause: Option<&Expr>,
    ) -> Result<ExecResult> {
        let schema = self.resolve_table_schema(table_name)?;
        let bound_where = match where_clause {
            Some(expr) => Some(bind_where_expr(&schema, expr)?),
            None => None,
        };

        let mut needed = vec![false; schema.columns.len()];
        if let Some(pk_idx) = schema.pk_col_idx {
            needed[pk_idx] = true;
        }
        let candidates = self.collect_matching_rows(&schema, bound_where.as_ref(), &needed)?;
        let mut to_delete: DeleteList = Vec::with_capacity(candidates.len());
        for (rid, row) in candidates {
            to_delete.push((rid, pk_of_row(&schema, &row)));
        }
        let affected = to_delete.len();
        if affected == 0 {
            return Ok(ExecResult::AffectedRows(0));
        }

        if let Some(txn) = &mut self.txn_mgr.active {
            let tws = txn.table_writes.entry(schema.table_id).or_default();
            for (rid, pk) in to_delete {
                if let Some(p) = pk {
                    tws.pk_adds.remove(&p);
                }
                if let Some((orig_committed_pk, _)) = tws.puts.remove(&rid) {
                    if let Some(opk) = orig_committed_pk {
                        tws.pk_removes.insert(opk);
                        tws.deletes.insert(rid, Some(opk));
                    } else if self
                        .storage
                        .tables
                        .get(&schema.table_id)
                        .is_some_and(|dt| dt.contains_row(rid))
                    {
                        tws.deletes.insert(rid, None);
                    }
                } else {
                    if let Some(p) = pk {
                        tws.pk_removes.insert(p);
                    }
                    tws.deletes.insert(rid, pk);
                }
            }
            return Ok(ExecResult::AffectedRows(affected));
        }

        self.commit_table_group(schema.table_id, Vec::new(), &to_delete)?;
        Ok(ExecResult::AffectedRows(affected))
    }

    fn exec_select(
        &mut self,
        table_name: &str,
        items: &[SelectItem],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        _search_mode: SearchMode,
    ) -> Result<ExecResult> {
        let schema = self.resolve_table_schema(table_name)?;
        let (bound_items, alias_map) = bind_select_items(&schema, items)?;
        let bound_where = match where_clause {
            Some(expr) => Some(bind_where_expr(&schema, expr)?),
            None => None,
        };
        let bound_order = bind_order_by(&schema, order_by, &alias_map)?;

        let columns: Vec<String> = bound_items
            .iter()
            .map(|it| match it {
                BoundSelectItem::Column { output_name, .. } => output_name.clone(),
                BoundSelectItem::L2Distance { output_name, .. } => output_name.clone(),
            })
            .collect();

        if limit == Some(0) {
            return Ok(ExecResult::Query {
                columns,
                rows: Vec::new(),
            });
        }

        let mut needed_cols = vec![false; schema.columns.len()];
        if let Some(pk_idx) = schema.pk_col_idx {
            needed_cols[pk_idx] = true;
        }
        for item in &bound_items {
            match item {
                BoundSelectItem::Column { col_idx, .. } => needed_cols[*col_idx] = true,
                BoundSelectItem::L2Distance { col_idx, .. } => needed_cols[*col_idx] = true,
            }
        }
        for (key, _) in &bound_order {
            match key {
                BoundOrderKey::Column(idx) => needed_cols[*idx] = true,
                BoundOrderKey::L2Distance { col_idx, .. } => needed_cols[*col_idx] = true,
            }
        }

        // Vector TopK (identical in `exact` and `ann` mode: always the exact answer).
        if bound_order.len() == 1 {
            if let (BoundOrderKey::L2Distance { col_idx, query }, true) = &bound_order[0] {
                if let Some(k) = limit {
                    let rows = self.exact_vector_topk(
                        &schema,
                        *col_idx,
                        query,
                        k,
                        bound_where.as_ref(),
                        &needed_cols,
                        &bound_items,
                    )?;
                    return Ok(ExecResult::Query { columns, rows });
                }
            }
        }

        // General ORDER BY + LIMIT K: stream with bounded max-heap of size K (O(K) RAM)
        if !bound_order.is_empty() {
            if let Some(k) = limit {
                return self.general_order_by_topk(
                    &schema,
                    bound_where.as_ref(),
                    &bound_order,
                    &needed_cols,
                    &bound_items,
                    &columns,
                    k,
                );
            }
        }

        if bound_order.is_empty() {
            let matched =
                self.collect_matching_rows_limited(&schema, bound_where.as_ref(), &needed_cols, limit)?;
            let mut out_rows = Vec::with_capacity(matched.len());
            for (_, row) in matched {
                out_rows.push(project_row(&bound_items, &row)?);
            }
            return Ok(ExecResult::Query {
                columns,
                rows: out_rows,
            });
        }

        let matched = self.collect_matching_rows(&schema, bound_where.as_ref(), &needed_cols)?;

        let mut keyed: Vec<(Vec<SortVal>, Option<i64>, RowId, Row)> =
            Vec::with_capacity(matched.len());
        for (rid, row) in matched {
            let mut keys = Vec::with_capacity(bound_order.len());
            for (key, _) in &bound_order {
                keys.push(eval_sort_key(key, &row)?);
            }
            let pk = pk_of_row(&schema, &row);
            keyed.push((keys, pk, rid, row));
        }
        keyed.sort_by(|a, b| compare_sort_entries(&a.0, a.1, a.2, &b.0, b.1, b.2, &bound_order));
        let mut out_rows = Vec::with_capacity(keyed.len());
        for (_, _, _, row) in keyed {
            out_rows.push(project_row(&bound_items, &row)?);
        }
        Ok(ExecResult::Query {
            columns,
            rows: out_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn general_order_by_topk(
        &self,
        schema: &TableSchema,
        where_expr: Option<&BoundExpr>,
        bound_order: &[(BoundOrderKey, bool)],
        needed_cols: &[bool],
        bound_items: &[BoundSelectItem],
        columns: &[String],
        k: usize,
    ) -> Result<ExecResult> {
        struct HeapEntry {
            keys: Vec<SortVal>,
            pk: Option<i64>,
            row_id: RowId,
            row: Row,
            dirs: std::rc::Rc<Vec<bool>>,
        }
        impl PartialEq for HeapEntry {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == Ordering::Equal
            }
        }
        impl Eq for HeapEntry {}
        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                for (i, &asc) in self.dirs.iter().enumerate() {
                    let ord = self.keys[i].cmp_val(&other.keys[i]);
                    if ord != Ordering::Equal {
                        return if asc { ord } else { ord.reverse() };
                    }
                }
                self.pk
                    .cmp(&other.pk)
                    .then_with(|| self.row_id.cmp(&other.row_id))
            }
        }

        let dirs: std::rc::Rc<Vec<bool>> =
            std::rc::Rc::new(bound_order.iter().map(|(_, asc)| *asc).collect());
        let mut scan_cols = vec![false; schema.columns.len()];
        if let Some(expr) = where_expr {
            collect_expr_cols(expr, &mut scan_cols);
        }
        if let Some(pk_idx) = schema.pk_col_idx {
            scan_cols[pk_idx] = true;
        }
        for (key, _) in bound_order {
            match key {
                BoundOrderKey::Column(idx) => scan_cols[*idx] = true,
                BoundOrderKey::L2Distance { col_idx, .. } => scan_cols[*col_idx] = true,
            }
        }

        let tid = schema.table_id;
        let tws_opt = self
            .txn_mgr
            .active
            .as_ref()
            .and_then(|t| t.table_writes.get(&tid));
        let mut heap: std::collections::BinaryHeap<HeapEntry> =
            std::collections::BinaryHeap::with_capacity(k.min(1 << 16).saturating_add(1));

        if let Some(dt) = self.storage.tables.get(&tid) {
            dt.scan_live(&scan_cols, |rid, row| {
                if let Some(tws) = tws_opt {
                    if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                        return Ok(true);
                    }
                }
                if let Some(expr) = where_expr {
                    if !eval_where(expr, &row)? {
                        return Ok(true);
                    }
                }
                let mut keys = Vec::with_capacity(bound_order.len());
                for (key, _) in bound_order {
                    keys.push(eval_sort_key(key, &row)?);
                }
                let pk = cell_pk(schema, &row);
                let accept = if heap.len() < k {
                    true
                } else if let Some(worst) = heap.peek() {
                    compare_sort_entries(&keys, pk, rid, &worst.keys, worst.pk, worst.row_id, bound_order)
                        == Ordering::Less
                } else {
                    true
                };
                if accept {
                    let entry = HeapEntry {
                        keys,
                        pk,
                        row_id: rid,
                        row: row.materialize(needed_cols)?,
                        dirs: dirs.clone(),
                    };
                    if heap.len() < k {
                        heap.push(entry);
                    } else if let Some(mut worst) = heap.peek_mut() {
                        *worst = entry;
                    }
                }
                Ok(true)
            })?;
        }

        if let Some(tws) = tws_opt {
            for (&rid, (_, row)) in &tws.puts {
                if let Some(expr) = where_expr {
                    if !eval_where(expr, row)? {
                        continue;
                    }
                }
                let mut keys = Vec::with_capacity(bound_order.len());
                for (key, _) in bound_order {
                    keys.push(eval_sort_key(key, row)?);
                }
                let pk = pk_of_row(schema, row);
                let entry = HeapEntry {
                    keys,
                    pk,
                    row_id: rid,
                    row: row.clone(),
                    dirs: dirs.clone(),
                };
                if heap.len() < k {
                    heap.push(entry);
                } else if let Some(mut worst) = heap.peek_mut() {
                    if entry < *worst {
                        *worst = entry;
                    }
                }
            }
        }

        let sorted = heap.into_sorted_vec();
        let mut out_rows = Vec::with_capacity(sorted.len());
        for entry in sorted {
            out_rows.push(project_row(bound_items, &entry.row)?);
        }
        Ok(ExecResult::Query {
            columns: columns.to_vec(),
            rows: out_rows,
        })
    }

    /// Exact TopK by `(distance, primary key, row id)` over the rows that pass `WHERE`
    /// and are visible to the current transaction.
    #[allow(clippy::too_many_arguments)]
    fn exact_vector_topk(
        &mut self,
        schema: &TableSchema,
        col_idx: usize,
        query: &[f32],
        k: usize,
        where_expr: Option<&BoundExpr>,
        needed_cols: &[bool],
        bound_items: &[BoundSelectItem],
    ) -> Result<Vec<Vec<JsonValue>>> {
        enum Winner<'a> {
            Slot(&'a MemTable, u32),
            Owned(Row),
            Txn(&'a Row),
        }

        let tid = schema.table_id;
        if let Some(dt) = self.storage.tables.get_mut(&tid) {
            dt.prepare_vector_search(col_idx);
        }
        let tws_opt = self
            .txn_mgr
            .active
            .as_ref()
            .and_then(|t| t.table_writes.get(&tid));

        let mut merged: Vec<(f64, Option<i64>, RowId, Winner<'_>)> = Vec::new();

        if let Some(dt) = self.storage.tables.get(&tid) {
            if let Some(mem) = dt.mem() {
                let vec_has_nulls = mem.vec_col(col_idx).map_or(true, |v| v.null_count > 0);
                let need_mask =
                    where_expr.is_some() || tws_opt.is_some() || mem.has_holes() || vec_has_nulls;
                let mask = if need_mask {
                    let mut m = mem.base_mask(Some(col_idx));
                    if let Some(expr) = where_expr {
                        apply_where_mask(mem, expr, &mut m)?;
                    }
                    if let Some(tws) = tws_opt {
                        for rid in tws.deletes.keys().chain(tws.puts.keys()) {
                            if let Some(slot) = dt.slot_of(*rid) {
                                m[slot as usize] = 0;
                            }
                        }
                    }
                    Some(m)
                } else {
                    None
                };
                for (d, pk, rid, slot) in mem.topk_exact(col_idx, query, k, mask.as_deref()) {
                    merged.push((d, pk, rid, Winner::Slot(mem, slot)));
                }
            } else {
                let mut access = vec![false; schema.columns.len()];
                if let Some(expr) = where_expr {
                    collect_expr_cols(expr, &mut access);
                }
                access[col_idx] = true;
                if let Some(pk_idx) = schema.pk_col_idx {
                    access[pk_idx] = true;
                }
                let mut heap = TopKHeap::new(k);
                dt.scan_live(&access, |rid, row| {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            return Ok(true);
                        }
                    }
                    if let Some(expr) = where_expr {
                        if !eval_where(expr, &row)? {
                            return Ok(true);
                        }
                    }
                    let Cell::Vector(v) = row.cell(col_idx) else {
                        return Ok(true);
                    };
                    let dist = l2_distance_squared(v, query)?;
                    let pk = cell_pk(schema, &row);
                    if heap.would_accept(dist, pk, rid) {
                        heap.push(TopKCandidate {
                            dist,
                            pk,
                            row_id: rid,
                            row: row.materialize(needed_cols)?,
                        });
                    }
                    Ok(true)
                })?;
                for c in heap.into_sorted_vec() {
                    merged.push((c.dist, c.pk, c.row_id, Winner::Owned(c.row)));
                }
            }
        }

        if let Some(tws) = tws_opt {
            for (&rid, (_, row)) in &tws.puts {
                if let Some(expr) = where_expr {
                    if !eval_where(expr, row)? {
                        continue;
                    }
                }
                let dist = match &row.values[col_idx] {
                    Value::Vector(v) => l2_distance_squared(v, query)?,
                    Value::Null => continue,
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "expected VECTOR column",
                        ));
                    }
                };
                merged.push((dist, pk_of_row(schema, row), rid, Winner::Txn(row)));
            }
            merged.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            merged.truncate(k);
        }

        let mut out_rows = Vec::with_capacity(merged.len());
        for (_, _, _, winner) in &merged {
            out_rows.push(match winner {
                Winner::Slot(mem, slot) => project_row(bound_items, &mem.row(*slot))?,
                Winner::Owned(row) => project_row(bound_items, row)?,
                Winner::Txn(row) => project_row(bound_items, *row)?,
            });
        }
        Ok(out_rows)
    }
}

#[inline]
fn cell_pk<R: RowAccess + ?Sized>(schema: &TableSchema, row: &R) -> Option<i64> {
    schema.pk_col_idx.and_then(|idx| match row.cell(idx) {
        Cell::Int(v) => Some(v),
        _ => None,
    })
}

/// If `expr` (a conjunction) pins the primary key to an integer literal, return it.
fn pk_eq_literal(schema: &TableSchema, expr: &BoundExpr) -> Option<i64> {
    let pk_idx = schema.pk_col_idx?;
    match expr {
        BoundExpr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => pk_eq_literal(schema, left).or_else(|| pk_eq_literal(schema, right)),
        BoundExpr::Binary {
            left,
            op: BinaryOp::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (BoundExpr::Column(c), BoundExpr::Literal(Value::BigInt(v)))
            | (BoundExpr::Literal(Value::BigInt(v)), BoundExpr::Column(c))
                if *c == pk_idx =>
            {
                Some(*v)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Clear the mask bit of every slot that does not satisfy `expr`.
/// Conjunctions of `column = literal` run as tight column loops.
fn apply_where_mask(mem: &MemTable, expr: &BoundExpr, mask: &mut [u8]) -> Result<()> {
    if let BoundExpr::Binary { left, op, right } = expr {
        match op {
            BinaryOp::And => {
                apply_where_mask(mem, left, mask)?;
                return apply_where_mask(mem, right, mask);
            }
            BinaryOp::Eq => match (left.as_ref(), right.as_ref()) {
                (BoundExpr::Column(c), BoundExpr::Literal(lit))
                | (BoundExpr::Literal(lit), BoundExpr::Column(c)) => {
                    if mem.and_eq_mask(*c, lit, mask) {
                        return Ok(());
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    for slot in 0..mask.len() {
        if mask[slot] != 0 && !eval_where(expr, &mem.row(slot as u32))? {
            mask[slot] = 0;
        }
    }
    Ok(())
}

/// `coerce_literal` for an owned literal: string payloads are moved, not copied.
pub fn coerce_literal_owned(lit: LiteralValue, target: &DataType, is_pk: bool) -> Result<Value> {
    if let (LiteralValue::Str(_), DataType::Varchar(max_len)) = (&lit, target) {
        let LiteralValue::Str(s) = lit else {
            unreachable!()
        };
        let chars = if s.is_ascii() { s.len() } else { s.chars().count() };
        if chars > *max_len {
            return Err(DbError::new(
                ErrorCode::ValueTooLong,
                format!("string length {} exceeds VARCHAR({})", chars, max_len),
            ));
        }
        return Ok(Value::Varchar(s));
    }
    coerce_literal(&lit, target, is_pk)
}

fn compare_sort_entries(
    ak: &[SortVal],
    apk: Option<i64>,
    arid: RowId,
    bk: &[SortVal],
    bpk: Option<i64>,
    brid: RowId,
    bound_order: &[(BoundOrderKey, bool)],
) -> Ordering {
    for (i, (_, asc)) in bound_order.iter().enumerate() {
        let ord = ak[i].cmp_val(&bk[i]);
        if ord != Ordering::Equal {
            return if *asc { ord } else { ord.reverse() };
        }
    }
    apk.cmp(&bpk).then_with(|| arid.cmp(&brid))
}

fn collect_expr_cols(expr: &BoundExpr, out: &mut [bool]) {
    match expr {
        BoundExpr::Column(idx) => out[*idx] = true,
        BoundExpr::L2Distance { col_idx, .. } => out[*col_idx] = true,
        BoundExpr::Not(inner) | BoundExpr::IsNull { expr: inner, .. } => {
            collect_expr_cols(inner, out);
        }
        BoundExpr::Binary { left, right, .. } => {
            collect_expr_cols(left, out);
            collect_expr_cols(right, out);
        }
        BoundExpr::Literal(_) | BoundExpr::FloatLiteral(_) => {}
    }
}

pub fn resolve_insert_columns(
    schema: &TableSchema,
    col_names: Option<&[String]>,
) -> Result<Vec<usize>> {
    match col_names {
        None => Ok((0..schema.columns.len()).collect()),
        Some(names) => {
            let mut indices = Vec::with_capacity(names.len());
            let mut seen = HashSet::new();
            for name in names {
                let (idx, _) = schema.find_column(name).ok_or_else(|| {
                    DbError::new(
                        ErrorCode::ColumnNotFound,
                        format!("column '{}' not found in table '{}'", name, schema.name),
                    )
                })?;
                if !seen.insert(idx) {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        format!("duplicate column '{}' in INSERT", name),
                    ));
                }
                indices.push(idx);
            }
            if let Some(pk_idx) = schema.pk_col_idx {
                if !seen.contains(&pk_idx) {
                    return Err(DbError::new(
                        ErrorCode::TypeMismatch,
                        "PRIMARY KEY column must be provided in INSERT",
                    ));
                }
            }
            Ok(indices)
        }
    }
}

pub fn coerce_literal(lit: &LiteralValue, target: &DataType, is_pk: bool) -> Result<Value> {
    if matches!(lit, LiteralValue::Null) {
        if is_pk {
            return Err(DbError::new(
                ErrorCode::TypeMismatch,
                "PRIMARY KEY column cannot be NULL",
            ));
        }
        return Ok(Value::Null);
    }
    match target {
        DataType::BigInt => match lit {
            LiteralValue::Int(i) => Ok(Value::BigInt(*i)),
            _ => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "expected BIGINT integer literal",
            )),
        },
        DataType::Varchar(max_len) => match lit {
            LiteralValue::Str(s) => {
                if s.chars().count() > *max_len {
                    return Err(DbError::new(
                        ErrorCode::ValueTooLong,
                        format!(
                            "string length {} exceeds VARCHAR({})",
                            s.chars().count(),
                            max_len
                        ),
                    ));
                }
                Ok(Value::Varchar(s.clone()))
            }
            _ => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "expected VARCHAR string literal",
            )),
        },
        DataType::Vector(dim) => {
            let vec = match lit {
                LiteralValue::Str(s) => parse_vector_str(s)?,
                LiteralValue::Vector(v) => validate_f64_slice_as_f32(v)?,
                _ => {
                    return Err(DbError::new(
                        ErrorCode::InvalidVector,
                        "expected VECTOR literal",
                    ));
                }
            };
            if vec.len() != *dim {
                return Err(DbError::new(
                    ErrorCode::DimensionMismatch,
                    format!(
                        "vector dimension mismatch: expected {}, got {}",
                        dim,
                        vec.len()
                    ),
                ));
            }
            Ok(Value::Vector(vec))
        }
    }
}

pub fn parse_query_vector(
    schema: &TableSchema,
    col_name: &str,
    q: &QueryVector,
) -> Result<(usize, Vec<f32>)> {
    let (col_idx, col_schema) = schema.find_column(col_name).ok_or_else(|| {
        DbError::new(
            ErrorCode::ColumnNotFound,
            format!("column '{}' not found in table '{}'", col_name, schema.name),
        )
    })?;
    let expected_dim = match col_schema.data_type {
        DataType::Vector(d) => d,
        _ => {
            return Err(DbError::new(
                ErrorCode::TypeMismatch,
                format!("column '{}' is not a VECTOR column", col_name),
            ));
        }
    };
    let vec = match q {
        QueryVector::Text(s) => parse_vector_str(s)?,
        QueryVector::Numbers(v) => validate_f64_slice_as_f32(v)?,
    };
    if vec.len() != expected_dim {
        return Err(DbError::new(
            ErrorCode::DimensionMismatch,
            format!(
                "query vector dimension mismatch: expected {}, got {}",
                expected_dim,
                vec.len()
            ),
        ));
    }
    Ok((col_idx, vec))
}

fn bind_select_items(
    schema: &TableSchema,
    items: &[SelectItem],
) -> Result<(Vec<BoundSelectItem>, HashMap<String, BoundOrderKey>)> {
    let mut out = Vec::new();
    let mut alias_map = HashMap::new();
    for item in items {
        match item {
            SelectItem::Wildcard => {
                for (idx, col) in schema.columns.iter().enumerate() {
                    out.push(BoundSelectItem::Column {
                        col_idx: idx,
                        output_name: col.name.clone(),
                    });
                }
            }
            SelectItem::Column { name, alias } => {
                let (idx, col) = schema.find_column(name).ok_or_else(|| {
                    DbError::new(
                        ErrorCode::ColumnNotFound,
                        format!("column '{}' not found in table '{}'", name, schema.name),
                    )
                })?;
                let output_name = alias.clone().unwrap_or_else(|| col.name.clone());
                if let Some(a) = alias {
                    alias_map.insert(a.to_ascii_lowercase(), BoundOrderKey::Column(idx));
                }
                out.push(BoundSelectItem::Column {
                    col_idx: idx,
                    output_name,
                });
            }
            SelectItem::L2Distance {
                column,
                query,
                alias,
            } => {
                let (col_idx, query_vec) = parse_query_vector(schema, column, query)?;
                let output_name = alias.clone().unwrap_or_else(|| "l2_distance".to_string());
                if let Some(a) = alias {
                    alias_map.insert(
                        a.to_ascii_lowercase(),
                        BoundOrderKey::L2Distance {
                            col_idx,
                            query: query_vec.clone(),
                        },
                    );
                }
                out.push(BoundSelectItem::L2Distance {
                    col_idx,
                    query: query_vec,
                    output_name,
                });
            }
        }
    }
    Ok((out, alias_map))
}

fn bind_order_by(
    schema: &TableSchema,
    items: &[OrderByItem],
    alias_map: &HashMap<String, BoundOrderKey>,
) -> Result<Vec<(BoundOrderKey, bool)>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let key = match &item.target {
            OrderByTarget::ColumnOrAlias(name) => {
                if let Some(aliased) = alias_map.get(&name.to_ascii_lowercase()) {
                    aliased.clone()
                } else {
                    let (idx, col) = schema.find_column(name).ok_or_else(|| {
                        DbError::new(
                            ErrorCode::ColumnNotFound,
                            format!("column or alias '{}' not found in ORDER BY", name),
                        )
                    })?;
                    if matches!(col.data_type, DataType::Vector(_)) {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "cannot ORDER BY raw VECTOR column directly",
                        ));
                    }
                    BoundOrderKey::Column(idx)
                }
            }
            OrderByTarget::L2Distance { column, query } => {
                let (col_idx, query_vec) = parse_query_vector(schema, column, query)?;
                BoundOrderKey::L2Distance {
                    col_idx,
                    query: query_vec,
                }
            }
        };
        out.push((key, item.asc));
    }
    Ok(out)
}

pub fn bind_where_expr(schema: &TableSchema, expr: &Expr) -> Result<BoundExpr> {
    let bound = bind_expr_inner(schema, expr)?;
    let ty = infer_expr_type(schema, &bound)?;
    if ty != ExprType::Bool {
        return Err(DbError::new(
            ErrorCode::TypeMismatch,
            "WHERE clause must evaluate to a boolean predicate",
        ));
    }
    Ok(bound)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExprType {
    Bool,
    BigInt,
    Varchar,
    Float,
    Vector,
    Null,
}

fn bind_expr_inner(schema: &TableSchema, expr: &Expr) -> Result<BoundExpr> {
    match expr {
        Expr::Column(name) => {
            let (idx, _) = schema.find_column(name).ok_or_else(|| {
                DbError::new(
                    ErrorCode::ColumnNotFound,
                    format!("column '{}' not found in table '{}'", name, schema.name),
                )
            })?;
            Ok(BoundExpr::Column(idx))
        }
        Expr::Literal(lit) => match lit {
            LiteralValue::Null => Ok(BoundExpr::Literal(Value::Null)),
            LiteralValue::Int(i) => Ok(BoundExpr::Literal(Value::BigInt(*i))),
            LiteralValue::Str(s) => Ok(BoundExpr::Literal(Value::Varchar(s.clone()))),
            LiteralValue::Vector(v) => Ok(BoundExpr::Literal(Value::Vector(
                validate_f64_slice_as_f32(v)?,
            ))),
            LiteralValue::Float(f) => {
                if !f.is_finite() {
                    return Err(DbError::new(
                        ErrorCode::InvalidArgument,
                        "non-finite float literal",
                    ));
                }
                Ok(BoundExpr::FloatLiteral(*f))
            }
        },
        Expr::L2Distance { column, query } => {
            let (col_idx, query_vec) = parse_query_vector(schema, column, query)?;
            Ok(BoundExpr::L2Distance {
                col_idx,
                query: query_vec,
            })
        }
        Expr::UnaryNot(inner) => {
            let b = bind_expr_inner(schema, inner)?;
            if infer_expr_type(schema, &b)? != ExprType::Bool {
                return Err(DbError::new(
                    ErrorCode::TypeMismatch,
                    "NOT requires boolean operand",
                ));
            }
            Ok(BoundExpr::Not(Box::new(b)))
        }
        Expr::IsNull { expr, negated } => {
            let b = bind_expr_inner(schema, expr)?;
            Ok(BoundExpr::IsNull {
                expr: Box::new(b),
                negated: *negated,
            })
        }
        Expr::Binary { left, op, right } => {
            let bl = bind_expr_inner(schema, left)?;
            let br = bind_expr_inner(schema, right)?;
            let tl = infer_expr_type(schema, &bl)?;
            let tr = infer_expr_type(schema, &br)?;
            match op {
                BinaryOp::And | BinaryOp::Or => {
                    if tl != ExprType::Bool || tr != ExprType::Bool {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "AND/OR operands must be boolean",
                        ));
                    }
                }
                BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::Lt
                | BinaryOp::Lte
                | BinaryOp::Gt
                | BinaryOp::Gte => {
                    let numeric_compat = (tl == ExprType::Float && tr == ExprType::BigInt)
                        || (tl == ExprType::BigInt && tr == ExprType::Float);
                    if tl != ExprType::Null && tr != ExprType::Null && tl != tr && !numeric_compat {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            format!("incompatible comparison types: {:?} vs {:?}", tl, tr),
                        ));
                    }
                    if matches!(tl, ExprType::Vector | ExprType::Bool)
                        || matches!(tr, ExprType::Vector | ExprType::Bool)
                    {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "cannot apply relational comparison to VECTOR or BOOLEAN",
                        ));
                    }
                }
            }
            Ok(BoundExpr::Binary {
                left: Box::new(bl),
                op: *op,
                right: Box::new(br),
            })
        }
    }
}

fn infer_expr_type(schema: &TableSchema, expr: &BoundExpr) -> Result<ExprType> {
    match expr {
        BoundExpr::Column(idx) => Ok(match schema.columns[*idx].data_type {
            DataType::BigInt => ExprType::BigInt,
            DataType::Varchar(_) => ExprType::Varchar,
            DataType::Vector(_) => ExprType::Vector,
        }),
        BoundExpr::Literal(v) => Ok(match v {
            Value::Null => ExprType::Null,
            Value::BigInt(_) => ExprType::BigInt,
            Value::Varchar(_) => ExprType::Varchar,
            Value::Vector(_) => ExprType::Vector,
        }),
        BoundExpr::FloatLiteral(_) | BoundExpr::L2Distance { .. } => Ok(ExprType::Float),
        BoundExpr::Not(_) | BoundExpr::IsNull { .. } => Ok(ExprType::Bool),
        BoundExpr::Binary { .. } => Ok(ExprType::Bool),
    }
}

#[inline]
pub fn eval_where<R: RowAccess + ?Sized>(expr: &BoundExpr, row: &R) -> Result<bool> {
    if let BoundExpr::Binary {
        left,
        op: BinaryOp::Eq,
        right,
    } = expr
    {
        match (left.as_ref(), right.as_ref()) {
            (BoundExpr::Column(cidx), BoundExpr::Literal(lit))
            | (BoundExpr::Literal(lit), BoundExpr::Column(cidx)) => {
                return Ok(match (row.cell(*cidx), lit) {
                    (Cell::Int(a), Value::BigInt(b)) => a == *b,
                    (Cell::Str(a), Value::Varchar(b)) => a == b.as_str(),
                    _ => false,
                });
            }
            _ => {}
        }
    }
    match eval_tri_bool(expr, row)? {
        Some(b) => Ok(b),
        None => Ok(false),
    }
}

fn eval_tri_bool<R: RowAccess + ?Sized>(expr: &BoundExpr, row: &R) -> Result<Option<bool>> {
    match expr {
        BoundExpr::Not(inner) => Ok(eval_tri_bool(inner, row)?.map(|b| !b)),
        BoundExpr::IsNull { expr, negated } => {
            let val = eval_scalar(expr, row)?;
            let is_null = matches!(val, EvalVal::Null);
            Ok(Some(if *negated { !is_null } else { is_null }))
        }
        BoundExpr::Binary { left, op, right } => match op {
            BinaryOp::And => {
                let l = eval_tri_bool(left, row)?;
                if l == Some(false) {
                    return Ok(Some(false));
                }
                let r = eval_tri_bool(right, row)?;
                match (l, r) {
                    (Some(false), _) | (_, Some(false)) => Ok(Some(false)),
                    (Some(true), Some(true)) => Ok(Some(true)),
                    _ => Ok(None),
                }
            }
            BinaryOp::Or => {
                let l = eval_tri_bool(left, row)?;
                if l == Some(true) {
                    return Ok(Some(true));
                }
                let r = eval_tri_bool(right, row)?;
                match (l, r) {
                    (Some(true), _) | (_, Some(true)) => Ok(Some(true)),
                    (Some(false), Some(false)) => Ok(Some(false)),
                    _ => Ok(None),
                }
            }
            _ => {
                let l = eval_scalar(left, row)?;
                let r = eval_scalar(right, row)?;
                let ord = match (l, r) {
                    (EvalVal::Null, _) | (_, EvalVal::Null) => return Ok(None),
                    (EvalVal::Int(a), EvalVal::Int(b)) => a.cmp(&b),
                    (EvalVal::Str(a), EvalVal::Str(b)) => a.cmp(b),
                    (EvalVal::Float(a), EvalVal::Float(b)) => a.total_cmp(&b),
                    (EvalVal::Float(a), EvalVal::Int(b)) => a.total_cmp(&(b as f64)),
                    (EvalVal::Int(a), EvalVal::Float(b)) => (a as f64).total_cmp(&b),
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "type mismatch during comparison",
                        ));
                    }
                };
                let res = match op {
                    BinaryOp::Eq => ord == Ordering::Equal,
                    BinaryOp::Neq => ord != Ordering::Equal,
                    BinaryOp::Lt => ord == Ordering::Less,
                    BinaryOp::Lte => ord != Ordering::Greater,
                    BinaryOp::Gt => ord == Ordering::Greater,
                    BinaryOp::Gte => ord != Ordering::Less,
                    _ => unreachable!(),
                };
                Ok(Some(res))
            }
        },
        _ => Err(DbError::new(
            ErrorCode::TypeMismatch,
            "expected boolean expression",
        )),
    }
}

enum EvalVal<'a> {
    Null,
    Int(i64),
    Str(&'a str),
    Float(f64),
}

fn eval_scalar<'a, R: RowAccess + ?Sized>(expr: &'a BoundExpr, row: &'a R) -> Result<EvalVal<'a>> {
    match expr {
        BoundExpr::Column(idx) => match row.cell(*idx) {
            Cell::Null => Ok(EvalVal::Null),
            Cell::Int(v) => Ok(EvalVal::Int(v)),
            Cell::Str(s) => Ok(EvalVal::Str(s)),
            Cell::Vector(_) => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "vector column cannot be compared directly",
            )),
        },
        BoundExpr::Literal(v) => match v {
            Value::Null => Ok(EvalVal::Null),
            Value::BigInt(i) => Ok(EvalVal::Int(*i)),
            Value::Varchar(s) => Ok(EvalVal::Str(s.as_str())),
            Value::Vector(_) => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "vector literal cannot be compared directly",
            )),
        },
        BoundExpr::FloatLiteral(f) => Ok(EvalVal::Float(*f)),
        BoundExpr::L2Distance { col_idx, query } => match row.cell(*col_idx) {
            Cell::Null => Ok(EvalVal::Null),
            Cell::Vector(v) => Ok(EvalVal::Float(l2_distance(v, query)?)),
            _ => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "L2_DISTANCE target is not a vector",
            )),
        },
        _ => Err(DbError::new(
            ErrorCode::TypeMismatch,
            "expected scalar expression",
        )),
    }
}

pub fn eval_sort_key<R: RowAccess + ?Sized>(key: &BoundOrderKey, row: &R) -> Result<SortVal> {
    match key {
        BoundOrderKey::Column(idx) => Ok(match row.cell(*idx) {
            Cell::Null => SortVal::Null,
            Cell::Int(v) => SortVal::Int(v),
            Cell::Str(s) => SortVal::Str(s.to_string()),
            Cell::Vector(_) => {
                return Err(DbError::new(
                    ErrorCode::TypeMismatch,
                    "cannot sort by VECTOR directly",
                ));
            }
        }),
        BoundOrderKey::L2Distance { col_idx, query } => match row.cell(*col_idx) {
            Cell::Null => Ok(SortVal::Null),
            Cell::Vector(v) => Ok(SortVal::Float(l2_distance(v, query)?)),
            _ => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "L2_DISTANCE column is not a VECTOR",
            )),
        },
    }
}

pub fn project_row<R: RowAccess + ?Sized>(items: &[BoundSelectItem], row: &R) -> Result<Vec<JsonValue>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            BoundSelectItem::Column { col_idx, .. } => {
                out.push(row.cell(*col_idx).to_json()?);
            }
            BoundSelectItem::L2Distance { col_idx, query, .. } => {
                let val = match row.cell(*col_idx) {
                    Cell::Null => JsonValue::Null,
                    Cell::Vector(v) => {
                        let dist = l2_distance(v, query)?;
                        let num = serde_json::Number::from_f64(dist).ok_or_else(|| {
                            DbError::new(
                                ErrorCode::InvalidVector,
                                "non-finite distance in output",
                            )
                        })?;
                        JsonValue::Number(num)
                    }
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "L2_DISTANCE column is not a VECTOR",
                        ));
                    }
                };
                out.push(val);
            }
        }
    }
    Ok(out)
}
