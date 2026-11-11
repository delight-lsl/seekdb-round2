pub mod catalog;
pub mod cli;
pub mod common;
pub mod recovery;
pub mod sql;
pub mod storage;
pub mod transaction;
pub mod vector;

use cli::protocol::DatabaseBackend;
use common::error::Result;
use common::types::{ExecResult, SearchMode};
use sql::executor::Database;

impl DatabaseBackend for Database {
    fn execute(&mut self, sql: &str, mode: SearchMode) -> Result<ExecResult> {
        self.execute_sql(sql, mode)
    }

    fn shutdown(&mut self) -> Result<()> {
        Database::shutdown(self)
    }
}
