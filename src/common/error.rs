use std::fmt;
use std::io;

/// Semantic error codes aligned with the MiniVectorDB evaluation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    UnsupportedSql,
    SyntaxError,
    TableExists,
    TableNotFound,
    ColumnNotFound,
    IndexExists,
    DuplicateKey,
    DimensionMismatch,
    InvalidVector,
    ValueTooLong,
    TypeMismatch,
    InvalidArgument,
    TransactionActive,
    NoActiveTransaction,
    IoError,
    InternalError,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnsupportedSql => "unsupported_sql",
            Self::SyntaxError => "syntax_error",
            Self::TableExists => "table_exists",
            Self::TableNotFound => "table_not_found",
            Self::ColumnNotFound => "column_not_found",
            Self::IndexExists => "index_exists",
            Self::DuplicateKey => "duplicate_key",
            Self::DimensionMismatch => "dimension_mismatch",
            Self::InvalidVector => "invalid_vector",
            Self::ValueTooLong => "value_too_long",
            Self::TypeMismatch => "type_mismatch",
            Self::InvalidArgument => "invalid_argument",
            Self::TransactionActive => "transaction_active",
            Self::NoActiveTransaction => "no_active_transaction",
            Self::IoError => "io_error",
            Self::InternalError => "internal_error",
        }
    }
}

/// Unified database error with semantic code and human-readable message.
#[derive(Debug, Clone)]
pub struct DbError {
    pub code: ErrorCode,
    pub message: String,
}

impl DbError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for DbError {}

impl From<io::Error> for DbError {
    fn from(err: io::Error) -> Self {
        Self::new(ErrorCode::IoError, format!("I/O error: {}", err))
    }
}

pub type Result<T> = std::result::Result<T, DbError>;
