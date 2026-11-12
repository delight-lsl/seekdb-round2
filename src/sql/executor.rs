use crate::catalog::manager::Catalog;
use crate::catalog::schema::TableSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{
    parse_vector_str, validate_f64_slice_as_f32, DataType, ExecResult, Row, RowId, SearchMode,
    TableId, Value,
};
use crate::recovery::checkpoint::{build_hnsw_from_disk, perform_checkpoint};
use crate::recovery::redo::recover_database;
use crate::recovery::wal::WalManager;
use crate::sql::ast::{
    BinaryOp, Expr, LiteralValue, OrderByItem, OrderByTarget, QueryVector, SelectItem, Statement,
};
use crate::sql::parser::Parser;
use crate::storage::codec::{
    decode_row_selective, encode_row, extract_pk_from_row_bytes, extract_vector_from_row_bytes,
};
use crate::storage::engine::{pk_of_row, StorageEngine};
use crate::transaction::manager::TransactionManager;
use crate::transaction::operation::WalOp;
use crate::vector::distance::{l2_distance, l2_distance_squared, l2_distance_squared_f32_fast};
use crate::vector::index::HnswIndex;
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

/// Complete MiniVectorDB database engine combining SQL execution, disk storage,
/// HNSW vector indexing, single-writer transactions, WAL, and crash recovery.
pub struct Database {
    pub data_dir: PathBuf,
    pub storage: StorageEngine,
    pub wal: WalManager,
    pub txn_mgr: TransactionManager,
    pub hnsw_indexes: HashMap<(TableId, usize), HnswIndex>,
}

impl Database {
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let (storage, wal, hnsw_indexes) = recover_database(&data_dir)?;
        Ok(Self {
            data_dir,
            storage,
            wal,
            txn_mgr: TransactionManager::new(),
            hnsw_indexes,
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
            } => self.exec_insert(&table_name, columns.as_deref(), &rows),
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

    pub fn shutdown(&mut self) -> Result<()> {
        self.txn_mgr.discard_if_active();
        let any_hnsw_dirty = self.hnsw_indexes.values().any(|idx| idx.is_dirty());
        let any_table_needs_compact = self.storage.tables.values().any(|dt| dt.needs_compaction());
        let has_ckp = self.data_dir.join("checkpoint.meta").exists();
        if !self.wal.dirty_since_checkpoint
            && !any_hnsw_dirty
            && !any_table_needs_compact
            && has_ckp
        {
            return Ok(());
        }
        for dt in self.storage.tables.values_mut() {
            dt.cache.clear();
            dt.cache.shrink_to_fit();
        }
        if !any_table_needs_compact {
            for hnsw in self.hnsw_indexes.values_mut() {
                hnsw.release_inmem_vectors_for_shutdown();
            }
        }
        perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        Ok(())
    }

    fn effective_catalog(&self) -> Catalog {
        if let Some(txn) = &self.txn_mgr.active {
            if txn.created_tables.is_empty() && txn.created_indexes.is_empty() {
                return self.storage.catalog.clone();
            }
            let mut cat = self.storage.catalog.clone();
            for schema in &txn.created_tables {
                let _ = cat.insert_table(schema.clone());
            }
            for (tid, idx) in &txn.created_indexes {
                let _ = cat.add_vector_index(*tid, idx.clone());
            }
            cat
        } else {
            self.storage.catalog.clone()
        }
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
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
        Ok(ExecResult::Ok)
    }

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
        self.storage
            .get_table_mut(tid)?
            .schema
            .indexes
            .push(idx_schema.clone());
        let hnsw = {
            let dt = self.storage.get_table(tid)?;
            build_hnsw_from_disk(
                &self.data_dir,
                dt,
                idx_schema.col_idx,
                idx_schema.dim,
                &idx_schema.name,
            )?
        };
        self.hnsw_indexes.insert((tid, idx_schema.col_idx), hnsw);
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
        Ok(ExecResult::Ok)
    }

    fn apply_index_updates_for_table(
        &mut self,
        tid: TableId,
        puts: &[(RowId, Option<i64>, Row)],
        deletes: &[(RowId, Option<i64>)],
    ) -> Result<()> {
        let schema = self.storage.get_table(tid)?.schema.clone();
        for idx_schema in &schema.indexes {
            if let Some(hnsw) = self.hnsw_indexes.get_mut(&(tid, idx_schema.col_idx)) {
                for &(rid, _) in deletes {
                    hnsw.delete(rid);
                }
                for (rid, _, row) in puts {
                    let pk = pk_of_row(&schema, row);
                    if let Value::Vector(ref vec) = row.values[idx_schema.col_idx] {
                        hnsw.upsert(*rid, pk, vec)?;
                    } else {
                        hnsw.delete(*rid);
                    }
                }
                if hnsw.needs_rebuild() {
                    let dt = self.storage.get_table(tid)?;
                    let rebuilt = build_hnsw_from_disk(
                        &self.data_dir,
                        dt,
                        idx_schema.col_idx,
                        idx_schema.dim,
                        &idx_schema.name,
                    )?;
                    self.hnsw_indexes.insert((tid, idx_schema.col_idx), rebuilt);
                }
            }
        }
        Ok(())
    }

    fn exec_commit(&mut self) -> Result<ExecResult> {
        let txn = self.txn_mgr.take_for_commit()?;
        let wal_ops = TransactionManager::build_wal_ops(&txn)?;
        if wal_ops.is_empty() {
            return Ok(ExecResult::Ok);
        }
        self.wal.append_committed_txn(&wal_ops)?;

        for schema in txn.created_tables {
            self.storage.create_table(schema)?;
        }
        let mut new_indexes = Vec::new();
        for (tid, idx) in txn.created_indexes {
            self.storage.catalog.add_vector_index(tid, idx.clone())?;
            self.storage
                .get_table_mut(tid)?
                .schema
                .indexes
                .push(idx.clone());
            new_indexes.push((tid, idx));
        }
        for (tid, tws) in txn.table_writes {
            let puts: Vec<(RowId, Option<i64>, Row)> = tws
                .puts
                .into_iter()
                .map(|(rid, (old_pk, row))| (rid, old_pk, row))
                .collect();
            let deletes: Vec<(RowId, Option<i64>)> = tws.deletes.into_iter().collect();
            self.apply_index_updates_for_table(tid, &puts, &deletes)?;
            {
                let dt = self.storage.get_table_mut(tid)?;
                dt.apply_batch_owned_with_wal(puts, &deletes, &wal_ops, false)?;
            }
        }
        for (tid, idx) in new_indexes {
            let dt = self.storage.get_table(tid)?;
            let hnsw = build_hnsw_from_disk(&self.data_dir, dt, idx.col_idx, idx.dim, &idx.name)?;
            self.hnsw_indexes.insert((tid, idx.col_idx), hnsw);
        }
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
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
        let base = self
            .storage
            .tables
            .get(&tid)
            .map_or(1, |dt| dt.next_row_id);
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
        raw_rows: &[Vec<LiteralValue>],
    ) -> Result<ExecResult> {
        let schema = self.resolve_table_schema(table_name)?;
        let col_mapping = resolve_insert_columns(&schema, col_names)?;

        let mut prepared_rows = Vec::with_capacity(raw_rows.len());
        let mut batch_pks = HashSet::new();

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
            for (val_idx, &col_idx) in col_mapping.iter().enumerate() {
                let col_schema = &schema.columns[col_idx];
                row_vals[col_idx] = coerce_literal(
                    &raw_vals[val_idx],
                    &col_schema.data_type,
                    col_schema.is_primary_key,
                )?;
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

        let mut puts = Vec::with_capacity(count);
        let mut wal_ops = Vec::with_capacity(count);
        for row in prepared_rows {
            let rid = next_rid;
            next_rid += 1;
            let mut row_bytes = Vec::new();
            encode_row(&row, &mut row_bytes)?;
            wal_ops.push(WalOp::PutRow {
                table_id: schema.table_id,
                row_id: rid,
                old_pk: None,
                row_bytes,
            });
            puts.push((rid, None, row));
        }

        self.wal.append_committed_txn(&wal_ops)?;
        self.apply_index_updates_for_table(schema.table_id, &puts, &[])?;
        {
            let dt = self.storage.get_table_mut(schema.table_id)?;
            dt.apply_batch_owned_with_wal(puts, &[], &wal_ops, false)?;
        }
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
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
            if dt.cache_covers_all() {
                for (rid, row) in dt.cache.iter() {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            continue;
                        }
                    }
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
            } else {
                dt.scan_live_streaming(&where_cols, |rid, payload, partial_row| {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            return Ok(true);
                        }
                    }
                    let ok = match where_expr {
                        Some(expr) => eval_where(expr, &partial_row)?,
                        None => true,
                    };
                    if ok {
                        let full_needed = decode_row_selective(payload, schema, needed_cols)?;
                        results.push((rid, full_needed));
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

        let mut updates: Vec<(RowId, Option<i64>, Row)> = Vec::with_capacity(candidates.len());
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

        let mut wal_ops = Vec::with_capacity(affected);
        for (rid, old_pk, row) in &updates {
            let mut row_bytes = Vec::new();
            encode_row(row, &mut row_bytes)?;
            wal_ops.push(WalOp::PutRow {
                table_id: schema.table_id,
                row_id: *rid,
                old_pk: *old_pk,
                row_bytes,
            });
        }
        self.wal.append_committed_txn(&wal_ops)?;
        self.apply_index_updates_for_table(schema.table_id, &updates, &[])?;
        {
            let dt = self.storage.get_table_mut(schema.table_id)?;
            dt.apply_batch_owned_with_wal(updates, &[], &wal_ops, false)?;
        }
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
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
        let mut to_delete = Vec::with_capacity(candidates.len());
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
                        .is_some_and(|dt| dt.row_locations.contains_key(&rid))
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

        let wal_ops: Vec<WalOp> = to_delete
            .iter()
            .map(|&(rid, old_pk)| WalOp::DeleteRow {
                table_id: schema.table_id,
                row_id: rid,
                old_pk,
            })
            .collect();
        self.wal.append_committed_txn(&wal_ops)?;
        {
            let dt = self.storage.get_table_mut(schema.table_id)?;
            dt.apply_batch(&[], &to_delete, false)?;
        }
        self.apply_index_updates_for_table(schema.table_id, &[], &to_delete)?;
        if self.wal.should_checkpoint() {
            perform_checkpoint(&mut self.storage, &mut self.wal, &mut self.hnsw_indexes, false)?;
        }
        Ok(ExecResult::AffectedRows(affected))
    }

    fn exec_select(
        &mut self,
        table_name: &str,
        items: &[SelectItem],
        where_clause: Option<&Expr>,
        order_by: &[OrderByItem],
        limit: Option<usize>,
        search_mode: SearchMode,
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

        if bound_order.len() == 1 {
            if let (BoundOrderKey::L2Distance { col_idx, query }, true) = &bound_order[0] {
                if let Some(k) = limit {
                    if search_mode == SearchMode::Ann
                        && self.hnsw_indexes.contains_key(&(schema.table_id, *col_idx))
                    {
                        if let Some(rows) = self.try_ann_topk(
                            &schema,
                            *col_idx,
                            query,
                            k,
                            bound_where.as_ref(),
                            &needed_cols,
                            &bound_items,
                        )? {
                            return Ok(ExecResult::Query { columns, rows });
                        }
                    }

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
        #[derive(Clone)]
        struct HeapEntry {
            keys: Vec<SortVal>,
            pk: Option<i64>,
            row_id: RowId,
            row: Row,
            dirs: Vec<bool>,
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

        let dirs: Vec<bool> = bound_order.iter().map(|(_, asc)| *asc).collect();
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
            std::collections::BinaryHeap::with_capacity(k.saturating_add(1));

        if let Some(dt) = self.storage.tables.get(&tid) {
            dt.scan_live_streaming(&scan_cols, |rid, payload, partial_row| {
                if let Some(tws) = tws_opt {
                    if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                        return Ok(true);
                    }
                }
                if let Some(expr) = where_expr {
                    if !eval_where(expr, &partial_row)? {
                        return Ok(true);
                    }
                }
                let mut keys = Vec::with_capacity(bound_order.len());
                for (key, _) in bound_order {
                    keys.push(eval_sort_key(key, &partial_row)?);
                }
                let pk = pk_of_row(schema, &partial_row);
                let accept = if heap.len() < k {
                    true
                } else if let Some(worst) = heap.peek() {
                    compare_sort_entries(&keys, pk, rid, &worst.keys, worst.pk, worst.row_id, bound_order)
                        == Ordering::Less
                } else {
                    true
                };
                if accept {
                    let full_row = decode_row_selective(payload, schema, needed_cols)?;
                    let entry = HeapEntry {
                        keys,
                        pk,
                        row_id: rid,
                        row: full_row,
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

    #[allow(clippy::too_many_arguments)]
    fn try_ann_topk(
        &mut self,
        schema: &TableSchema,
        col_idx: usize,
        query: &[f32],
        k: usize,
        where_expr: Option<&BoundExpr>,
        needed_cols: &[bool],
        bound_items: &[BoundSelectItem],
    ) -> Result<Option<Vec<Vec<JsonValue>>>> {
        let tid = schema.table_id;
        let tws_opt = self
            .txn_mgr
            .active
            .as_ref()
            .and_then(|t| t.table_writes.get(&tid));
        let mut tx_excluded = HashSet::new();
        if let Some(tws) = tws_opt {
            for &rid in tws.deletes.keys() {
                tx_excluded.insert(rid);
            }
            for &rid in tws.puts.keys() {
                tx_excluded.insert(rid);
            }
        }

        let dt_ref = self.storage.tables.get(&tid);
        let hnsw = self.hnsw_indexes.get_mut(&(tid, col_idx)).unwrap();

        let (cands, exhaustive) = if let Some(w_expr) = where_expr {
            let Some(dt) = dt_ref else {
                return Ok(None);
            };
            let mut where_cols = vec![false; schema.columns.len()];
            collect_expr_cols(w_expr, &mut where_cols);
            let mut filter_fn = |rid: RowId| -> Result<bool> {
                if let Some(cached) = dt.cache.get(rid) {
                    return eval_where(w_expr, cached);
                }
                let &(off, flen) = dt.row_locations.get(&rid).ok_or_else(|| {
                    DbError::new(ErrorCode::InternalError, "HNSW row_id missing in DiskTable")
                })?;
                let frame = dt.file.read_frame_at(off, flen)?;
                let partial = decode_row_selective(&frame.payload, schema, &where_cols)?;
                eval_where(w_expr, &partial)
            };
            hnsw.search_knn(query, k, &tx_excluded, Some(&mut filter_fn))?
        } else {
            hnsw.search_knn(query, k, &tx_excluded, None::<&mut fn(RowId) -> Result<bool>>)?
        };

        if cands.len() < k && !exhaustive {
            return Ok(None);
        }

        if tws_opt.is_none() {
            if let Some(dt) = dt_ref {
                let mut out_rows = Vec::with_capacity(cands.len());
                for (_, _, rid) in cands {
                    if let Some(cached) = dt.cache.get(rid) {
                        out_rows.push(project_row(bound_items, cached)?);
                    } else {
                        let &(off, flen) = dt.row_locations.get(&rid).ok_or_else(|| {
                            DbError::new(
                                ErrorCode::InternalError,
                                "HNSW row_id missing in DiskTable",
                            )
                        })?;
                        let frame = dt.file.read_frame_at(off, flen)?;
                        let row = decode_row_selective(&frame.payload, schema, needed_cols)?;
                        out_rows.push(project_row(bound_items, &row)?);
                    }
                }
                return Ok(Some(out_rows));
            }
        }

        let mut heap = TopKHeap::new(k);
        if let Some(dt) = dt_ref {
            for (dist_sq, pk, rid) in cands {
                if !heap.would_accept(dist_sq, pk, rid) {
                    continue;
                }
                let row = if let Some(cached) = dt.cache.get(rid) {
                    cached.clone()
                } else {
                    let &(off, flen) = dt.row_locations.get(&rid).ok_or_else(|| {
                        DbError::new(ErrorCode::InternalError, "HNSW row_id missing in DiskTable")
                    })?;
                    let frame = dt.file.read_frame_at(off, flen)?;
                    decode_row_selective(&frame.payload, schema, needed_cols)?
                };
                heap.push(TopKCandidate {
                    dist_sq,
                    pk,
                    row_id: rid,
                    row,
                });
            }
        }

        if let Some(tws) = tws_opt {
            for (&rid, (_, row)) in &tws.puts {
                if let Some(w_expr) = where_expr {
                    if !eval_where(w_expr, row)? {
                        continue;
                    }
                }
                let dist_sq = match &row.values[col_idx] {
                    Value::Vector(v) => l2_distance_squared(v, query)?,
                    Value::Null => continue,
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "expected VECTOR column",
                        ));
                    }
                };
                let pk = pk_of_row(schema, row);
                if heap.would_accept(dist_sq, pk, rid) {
                    heap.push(TopKCandidate {
                        dist_sq,
                        pk,
                        row_id: rid,
                        row: row.clone(),
                    });
                }
            }
        }

        let winners = heap.into_sorted_vec();
        let mut out_rows = Vec::with_capacity(winners.len());
        for cand in winners {
            out_rows.push(project_row(bound_items, &cand.row)?);
        }
        Ok(Some(out_rows))
    }

    #[allow(clippy::too_many_arguments)]
    fn exact_vector_topk(
        &self,
        schema: &TableSchema,
        col_idx: usize,
        query: &[f32],
        k: usize,
        where_expr: Option<&BoundExpr>,
        needed_cols: &[bool],
        bound_items: &[BoundSelectItem],
    ) -> Result<Vec<Vec<JsonValue>>> {
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

        let use_fast_f32 = query.iter().all(|&x| x.abs() < 1e15)
            && self
                .hnsw_indexes
                .get(&(tid, col_idx))
                .is_some_and(|h| h.is_safe_for_f32_dist());

        let mut heap = TopKHeap::new(k);

        if let Some(dt) = self.storage.tables.get(&tid) {
            if dt.cache_covers_all() {
                // Zero-clone fast path: hold only lightweight (dist_sq, pk, rid) in heap,
                // then resolve winner Rows at the end.
                #[derive(Clone, Copy)]
                struct LightCand {
                    dist_sq: f64,
                    pk: Option<i64>,
                    row_id: RowId,
                }
                impl PartialEq for LightCand {
                    fn eq(&self, other: &Self) -> bool {
                        self.dist_sq.to_bits() == other.dist_sq.to_bits()
                            && self.pk == other.pk
                            && self.row_id == other.row_id
                    }
                }
                impl Eq for LightCand {}
                impl PartialOrd for LightCand {
                    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                        Some(self.cmp(other))
                    }
                }
                impl Ord for LightCand {
                    fn cmp(&self, other: &Self) -> Ordering {
                        self.dist_sq
                            .total_cmp(&other.dist_sq)
                            .then_with(|| self.pk.cmp(&other.pk))
                            .then_with(|| self.row_id.cmp(&other.row_id))
                    }
                }
                let mut light_heap: std::collections::BinaryHeap<LightCand> =
                    std::collections::BinaryHeap::with_capacity(k.saturating_add(1));
                for (rid, row) in dt.cache.iter() {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            continue;
                        }
                    }
                    if let Some(w_expr) = where_expr {
                        if !eval_where(w_expr, row)? {
                            continue;
                        }
                    }
                    let dist_sq = match &row.values[col_idx] {
                        Value::Vector(v) => {
                            if use_fast_f32 {
                                l2_distance_squared_f32_fast(v, query) as f64
                            } else {
                                l2_distance_squared(v, query)?
                            }
                        }
                        Value::Null => continue,
                        _ => {
                            return Err(DbError::new(
                                ErrorCode::TypeMismatch,
                                "expected VECTOR column",
                            ));
                        }
                    };
                    let pk = pk_of_row(schema, row);
                    let cand = LightCand {
                        dist_sq,
                        pk,
                        row_id: rid,
                    };
                    if light_heap.len() < k {
                        light_heap.push(cand);
                    } else if let Some(mut worst) = light_heap.peek_mut() {
                        if cand < *worst {
                            *worst = cand;
                        }
                    }
                }
                if tws_opt.is_none() {
                    let sorted = light_heap.into_sorted_vec();
                    let mut out_rows = Vec::with_capacity(sorted.len());
                    for cand in sorted {
                        if let Some(row) = dt.cache.get(cand.row_id) {
                            out_rows.push(project_row(bound_items, row)?);
                        }
                    }
                    return Ok(out_rows);
                }
                for cand in light_heap {
                    if let Some(row) = dt.cache.get(cand.row_id) {
                        heap.push(TopKCandidate {
                            dist_sq: cand.dist_sq,
                            pk: cand.pk,
                            row_id: cand.row_id,
                            row: row.clone(),
                        });
                    }
                }
            } else {
                dt.scan_live_streaming(&where_cols, |rid, payload, partial_row| {
                    if let Some(tws) = tws_opt {
                        if tws.deletes.contains_key(&rid) || tws.puts.contains_key(&rid) {
                            return Ok(true);
                        }
                    }
                    if let Some(w_expr) = where_expr {
                        if !eval_where(w_expr, &partial_row)? {
                            return Ok(true);
                        }
                    }
                    let vec_opt = if where_cols[col_idx] {
                        match &partial_row.values[col_idx] {
                            Value::Vector(v) => Some(v.clone()),
                            Value::Null => None,
                            _ => None,
                        }
                    } else {
                        extract_vector_from_row_bytes(payload, col_idx)?
                    };
                    let Some(vec) = vec_opt else {
                        return Ok(true);
                    };
                    let dist_sq = l2_distance_squared(&vec, query)?;
                    let pk = match schema.pk_col_idx {
                        Some(pk_idx) => {
                            if where_cols[pk_idx] {
                                pk_of_row(schema, &partial_row)
                            } else {
                                extract_pk_from_row_bytes(payload, pk_idx)?
                            }
                        }
                        None => None,
                    };
                    if heap.would_accept(dist_sq, pk, rid) {
                        let full_row = decode_row_selective(payload, schema, needed_cols)?;
                        heap.push(TopKCandidate {
                            dist_sq,
                            pk,
                            row_id: rid,
                            row: full_row,
                        });
                    }
                    Ok(true)
                })?;
            }
        }

        if let Some(tws) = tws_opt {
            for (&rid, (_, row)) in &tws.puts {
                if let Some(w_expr) = where_expr {
                    if !eval_where(w_expr, row)? {
                        continue;
                    }
                }
                let dist_sq = match &row.values[col_idx] {
                    Value::Vector(v) => l2_distance_squared(v, query)?,
                    Value::Null => continue,
                    _ => {
                        return Err(DbError::new(
                            ErrorCode::TypeMismatch,
                            "expected VECTOR column",
                        ));
                    }
                };
                let pk = pk_of_row(schema, row);
                if heap.would_accept(dist_sq, pk, rid) {
                    heap.push(TopKCandidate {
                        dist_sq,
                        pk,
                        row_id: rid,
                        row: row.clone(),
                    });
                }
            }
        }

        let winners = heap.into_sorted_vec();
        let mut out_rows = Vec::with_capacity(winners.len());
        for cand in winners {
            out_rows.push(project_row(bound_items, &cand.row)?);
        }
        Ok(out_rows)
    }
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

pub fn eval_where(expr: &BoundExpr, row: &Row) -> Result<bool> {
    match eval_tri_bool(expr, row)? {
        Some(b) => Ok(b),
        None => Ok(false),
    }
}

fn eval_tri_bool(expr: &BoundExpr, row: &Row) -> Result<Option<bool>> {
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

fn eval_scalar<'a>(expr: &'a BoundExpr, row: &'a Row) -> Result<EvalVal<'a>> {
    match expr {
        BoundExpr::Column(idx) => match &row.values[*idx] {
            Value::Null => Ok(EvalVal::Null),
            Value::BigInt(v) => Ok(EvalVal::Int(*v)),
            Value::Varchar(s) => Ok(EvalVal::Str(s.as_str())),
            Value::Vector(_) => Err(DbError::new(
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
        BoundExpr::L2Distance { col_idx, query } => match &row.values[*col_idx] {
            Value::Null => Ok(EvalVal::Null),
            Value::Vector(v) => Ok(EvalVal::Float(l2_distance(v, query)?)),
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

pub fn eval_sort_key(key: &BoundOrderKey, row: &Row) -> Result<SortVal> {
    match key {
        BoundOrderKey::Column(idx) => Ok(match &row.values[*idx] {
            Value::Null => SortVal::Null,
            Value::BigInt(v) => SortVal::Int(*v),
            Value::Varchar(s) => SortVal::Str(s.clone()),
            Value::Vector(_) => {
                return Err(DbError::new(
                    ErrorCode::TypeMismatch,
                    "cannot sort by VECTOR directly",
                ));
            }
        }),
        BoundOrderKey::L2Distance { col_idx, query } => match &row.values[*col_idx] {
            Value::Null => Ok(SortVal::Null),
            Value::Vector(v) => Ok(SortVal::Float(l2_distance_squared(v, query)?)),
            _ => Err(DbError::new(
                ErrorCode::TypeMismatch,
                "L2_DISTANCE column is not a VECTOR",
            )),
        },
    }
}

pub fn project_row(items: &[BoundSelectItem], row: &Row) -> Result<Vec<JsonValue>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            BoundSelectItem::Column { col_idx, .. } => {
                out.push(row.values[*col_idx].to_json()?);
            }
            BoundSelectItem::L2Distance { col_idx, query, .. } => {
                let val = match &row.values[*col_idx] {
                    Value::Null => JsonValue::Null,
                    Value::Vector(v) => {
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
