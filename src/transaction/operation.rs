use crate::catalog::schema::{TableSchema, VectorIndexSchema};
use crate::common::types::{Row, RowId, TableId};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Uncommitted per-table changes buffered in an active explicit transaction.
#[derive(Debug, Clone, Default)]
pub struct TableTxnWriteSet {
    pub puts: BTreeMap<RowId, (Option<i64>, Row)>,
    pub deletes: HashMap<RowId, Option<i64>>,
    pub pk_adds: HashMap<i64, RowId>,
    pub pk_removes: HashSet<i64>,
    pub next_row_id: Option<RowId>,
}

/// Active single-writer explicit transaction state.
#[derive(Debug, Clone, Default)]
pub struct TxnWriteSet {
    pub created_tables: Vec<TableSchema>,
    pub created_indexes: Vec<(TableId, VectorIndexSchema)>,
    pub table_writes: HashMap<TableId, TableTxnWriteSet>,
}

/// Redo-log operation emitted on commit (both explicit and implicit transactions).
#[derive(Debug, Clone)]
pub enum WalOp {
    CreateTable(TableSchema),
    CreateIndex {
        table_id: TableId,
        index: VectorIndexSchema,
    },
    PutRow {
        table_id: TableId,
        row_id: RowId,
        old_pk: Option<i64>,
        row_bytes: Vec<u8>,
    },
    DeleteRow {
        table_id: TableId,
        row_id: RowId,
        old_pk: Option<i64>,
    },
}
