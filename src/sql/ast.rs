use crate::catalog::schema::ColumnSchema;

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Null,
    Int(i64),
    Float(f64),
    Str(String),
    Vector(Vec<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(LiteralValue),
    L2Distance {
        column: String,
        query: QueryVector,
    },
    UnaryNot(Box<Expr>),
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryVector {
    Text(String),
    Numbers(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    Wildcard,
    Column {
        name: String,
        alias: Option<String>,
    },
    L2Distance {
        column: String,
        query: QueryVector,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderByTarget {
    ColumnOrAlias(String),
    L2Distance {
        column: String,
        query: QueryVector,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem {
    pub target: OrderByTarget,
    pub asc: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    CreateTable {
        name: String,
        columns: Vec<ColumnSchema>,
    },
    CreateVectorIndex {
        index_name: String,
        table_name: String,
        column_name: String,
    },
    Insert {
        table_name: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<LiteralValue>>,
    },
    Select {
        items: Vec<SelectItem>,
        table_name: String,
        where_clause: Option<Expr>,
        order_by: Vec<OrderByItem>,
        limit: Option<usize>,
    },
    Update {
        table_name: String,
        assignments: Vec<(String, LiteralValue)>,
        where_clause: Option<Expr>,
    },
    Delete {
        table_name: String,
        where_clause: Option<Expr>,
    },
    Begin,
    Commit,
    Rollback,
}
