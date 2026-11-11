use crate::common::error::{DbError, ErrorCode, Result};
use crate::storage::codec::encode_row;
use crate::transaction::operation::{TxnWriteSet, WalOp};

/// Single-client transaction lifecycle manager.
#[derive(Debug, Default)]
pub struct TransactionManager {
    pub active: Option<TxnWriteSet>,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self { active: None }
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin(&mut self) -> Result<()> {
        if self.active.is_some() {
            return Err(DbError::new(
                ErrorCode::TransactionActive,
                "a transaction is already active",
            ));
        }
        self.active = Some(TxnWriteSet::default());
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<()> {
        if self.active.take().is_none() {
            return Err(DbError::new(
                ErrorCode::NoActiveTransaction,
                "no active transaction to rollback",
            ));
        }
        Ok(())
    }

    pub fn take_for_commit(&mut self) -> Result<TxnWriteSet> {
        self.active.take().ok_or_else(|| {
            DbError::new(
                ErrorCode::NoActiveTransaction,
                "no active transaction to commit",
            )
        })
    }

    pub fn discard_if_active(&mut self) {
        self.active = None;
    }

    pub fn build_wal_ops(txn: &TxnWriteSet) -> Result<Vec<WalOp>> {
        let mut ops = Vec::new();
        for schema in &txn.created_tables {
            ops.push(WalOp::CreateTable(schema.clone()));
        }
        for (tid, idx) in &txn.created_indexes {
            ops.push(WalOp::CreateIndex {
                table_id: *tid,
                index: idx.clone(),
            });
        }
        for (&tid, tws) in &txn.table_writes {
            for (&rid, (old_pk, row)) in &tws.puts {
                let mut row_bytes = Vec::new();
                encode_row(row, &mut row_bytes)?;
                ops.push(WalOp::PutRow {
                    table_id: tid,
                    row_id: rid,
                    old_pk: *old_pk,
                    row_bytes,
                });
            }
            for (&rid, &old_pk) in &tws.deletes {
                ops.push(WalOp::DeleteRow {
                    table_id: tid,
                    row_id: rid,
                    old_pk,
                });
            }
        }
        Ok(ops)
    }
}
