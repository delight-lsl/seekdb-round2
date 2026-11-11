use crate::common::types::{DataType, TableId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: DataType,
    pub is_primary_key: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorIndexSchema {
    pub name: String,
    pub column_name: String,
    pub col_idx: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_id: TableId,
    pub name: String,
    pub columns: Vec<ColumnSchema>,
    pub pk_col_idx: Option<usize>,
    pub indexes: Vec<VectorIndexSchema>,
}

impl TableSchema {
    pub fn find_column(&self, name: &str) -> Option<(usize, &ColumnSchema)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name.eq_ignore_ascii_case(name))
    }

    pub fn find_vector_index_by_col(&self, col_idx: usize) -> Option<&VectorIndexSchema> {
        self.indexes.iter().find(|idx| idx.col_idx == col_idx)
    }

    pub fn has_index_name(&self, name: &str) -> bool {
        self.indexes
            .iter()
            .any(|idx| idx.name.eq_ignore_ascii_case(name))
    }
}
