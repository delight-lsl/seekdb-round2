use crate::common::error::{DbError, ErrorCode, Result};
use crate::transaction::operation::TxnWriteSet;

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
}
