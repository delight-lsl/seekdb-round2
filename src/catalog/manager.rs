use crate::catalog::schema::{ColumnSchema, TableSchema, VectorIndexSchema};
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{DataType, TableId};
use std::collections::HashMap;

/// Manages committed table and vector index metadata.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    tables: HashMap<String, TableSchema>,
    id_to_name: HashMap<TableId, String>,
    next_table_id: TableId,
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            id_to_name: HashMap::new(),
            next_table_id: 1,
        }
    }

    pub fn from_state(tables: Vec<TableSchema>, next_table_id: TableId) -> Self {
        let mut cat = Self {
            tables: HashMap::new(),
            id_to_name: HashMap::new(),
            next_table_id: next_table_id.max(1),
        };
        for schema in tables {
            let key = schema.name.to_ascii_lowercase();
            cat.next_table_id = cat.next_table_id.max(schema.table_id + 1);
            cat.id_to_name.insert(schema.table_id, key.clone());
            cat.tables.insert(key, schema);
        }
        cat
    }

    pub fn next_table_id(&self) -> TableId {
        self.next_table_id
    }

    pub fn bump_next_table_id(&mut self, at_least: TableId) {
        self.next_table_id = self.next_table_id.max(at_least).max(1);
    }

    /// Forget a table that never reached its commit point.
    pub fn remove_table(&mut self, table_id: TableId) {
        if let Some(key) = self.id_to_name.remove(&table_id) {
            self.tables.remove(&key);
        }
    }

    pub fn all_tables(&self) -> impl Iterator<Item = &TableSchema> {
        self.tables.values()
    }

    pub fn get_table(&self, name: &str) -> Result<&TableSchema> {
        self.tables
            .get(&name.to_ascii_lowercase())
            .ok_or_else(|| {
                DbError::new(
                    ErrorCode::TableNotFound,
                    format!("table '{}' does not exist", name),
                )
            })
    }

    pub fn get_table_mut(&mut self, name: &str) -> Result<&mut TableSchema> {
        self.tables
            .get_mut(&name.to_ascii_lowercase())
            .ok_or_else(|| {
                DbError::new(
                    ErrorCode::TableNotFound,
                    format!("table '{}' does not exist", name),
                )
            })
    }

    pub fn get_table_by_id(&self, table_id: TableId) -> Option<&TableSchema> {
        self.id_to_name
            .get(&table_id)
            .and_then(|name| self.tables.get(name))
    }

    pub fn validate_new_table(
        &self,
        name: &str,
        columns: Vec<ColumnSchema>,
    ) -> Result<TableSchema> {
        let key = name.to_ascii_lowercase();
        if self.tables.contains_key(&key) {
            return Err(DbError::new(
                ErrorCode::TableExists,
                format!("table '{}' already exists", name),
            ));
        }
        if columns.is_empty() {
            return Err(DbError::new(
                ErrorCode::SyntaxError,
                "table must have at least one column",
            ));
        }
        let mut seen_cols = std::collections::HashSet::new();
        let mut pk_col_idx = None;
        for (i, col) in columns.iter().enumerate() {
            let col_key = col.name.to_ascii_lowercase();
            if !seen_cols.insert(col_key) {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    format!("duplicate column name '{}'", col.name),
                ));
            }
            if let DataType::Vector(dim) = col.data_type {
                if dim == 0 {
                    return Err(DbError::new(
                        ErrorCode::InvalidVector,
                        "VECTOR dimension must be positive",
                    ));
                }
            }
            if col.is_primary_key {
                if pk_col_idx.is_some() {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "multiple primary keys are not supported",
                    ));
                }
                if col.data_type != DataType::BigInt {
                    return Err(DbError::new(
                        ErrorCode::UnsupportedSql,
                        "PRIMARY KEY is only supported on BIGINT columns",
                    ));
                }
                pk_col_idx = Some(i);
            }
        }
        Ok(TableSchema {
            table_id: self.next_table_id,
            name: name.to_string(),
            columns,
            pk_col_idx,
            indexes: Vec::new(),
        })
    }

    pub fn insert_table(&mut self, schema: TableSchema) -> Result<()> {
        let key = schema.name.to_ascii_lowercase();
        if self.tables.contains_key(&key) {
            return Err(DbError::new(
                ErrorCode::TableExists,
                format!("table '{}' already exists", schema.name),
            ));
        }
        if schema.table_id >= self.next_table_id {
            self.next_table_id = schema.table_id + 1;
        }
        self.id_to_name.insert(schema.table_id, key.clone());
        self.tables.insert(key, schema);
        Ok(())
    }

    pub fn validate_new_vector_index(
        &self,
        index_name: &str,
        table_name: &str,
        column_name: &str,
    ) -> Result<(TableId, VectorIndexSchema)> {
        // Ensure index name is unique across all tables
        for tbl in self.tables.values() {
            if tbl.has_index_name(index_name) {
                return Err(DbError::new(
                    ErrorCode::IndexExists,
                    format!("vector index '{}' already exists", index_name),
                ));
            }
        }
        let tbl = self.get_table(table_name)?;
        let (col_idx, col) = tbl.find_column(column_name).ok_or_else(|| {
            DbError::new(
                ErrorCode::ColumnNotFound,
                format!("column '{}' not found in table '{}'", column_name, table_name),
            )
        })?;
        let dim = match col.data_type {
            DataType::Vector(d) => d,
            _ => {
                return Err(DbError::new(
                    ErrorCode::TypeMismatch,
                    format!("column '{}' is not a VECTOR column", column_name),
                ));
            }
        };
        if tbl.find_vector_index_by_col(col_idx).is_some() {
            return Err(DbError::new(
                ErrorCode::IndexExists,
                format!(
                    "column '{}' on table '{}' already has a vector index",
                    column_name, table_name
                ),
            ));
        }
        Ok((
            tbl.table_id,
            VectorIndexSchema {
                name: index_name.to_string(),
                column_name: col.name.clone(),
                col_idx,
                dim,
            },
        ))
    }

    pub fn add_vector_index(
        &mut self,
        table_id: TableId,
        index: VectorIndexSchema,
    ) -> Result<()> {
        let key = self
            .id_to_name
            .get(&table_id)
            .cloned()
            .ok_or_else(|| DbError::new(ErrorCode::TableNotFound, "table id not found"))?;
        let tbl = self
            .tables
            .get_mut(&key)
            .ok_or_else(|| DbError::new(ErrorCode::TableNotFound, "table not found"))?;
        if tbl.has_index_name(&index.name) || tbl.find_vector_index_by_col(index.col_idx).is_some() {
            return Err(DbError::new(
                ErrorCode::IndexExists,
                format!("vector index '{}' already exists", index.name),
            ));
        }
        tbl.indexes.push(index);
        Ok(())
    }
}
