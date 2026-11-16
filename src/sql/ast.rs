use crate::catalog::schema::ColumnSchema;
use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue<'a> {
    Null,
    Int(i64),
    Float(f64),
    Str(Cow<'a, str>),
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
pub enum Expr<'a> {
    Column(String),
    Literal(LiteralValue<'a>),
    L2Distance {
        column: String,
        query: QueryVector<'a>,
    },
    UnaryNot(Box<Expr<'a>>),
    IsNull {
        expr: Box<Expr<'a>>,
        negated: bool,
    },
    Binary {
        left: Box<Expr<'a>>,
        op: BinaryOp,
        right: Box<Expr<'a>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryVector<'a> {
    Text(Cow<'a, str>),
    Numbers(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem<'a> {
    Wildcard,
    Column {
        name: String,
        alias: Option<String>,
    },
    L2Distance {
        column: String,
        query: QueryVector<'a>,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderByTarget<'a> {
    ColumnOrAlias(String),
    L2Distance {
        column: String,
        query: QueryVector<'a>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem<'a> {
    pub target: OrderByTarget<'a>,
    pub asc: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement<'a> {
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
        rows: Vec<Vec<LiteralValue<'a>>>,
    },
    Select {
        items: Vec<SelectItem<'a>>,
        table_name: String,
        where_clause: Option<Expr<'a>>,
        order_by: Vec<OrderByItem<'a>>,
        limit: Option<usize>,
    },
    Update {
        table_name: String,
        assignments: Vec<(String, LiteralValue<'a>)>,
        where_clause: Option<Expr<'a>>,
    },
    Delete {
        table_name: String,
        where_clause: Option<Expr<'a>>,
    },
    Begin,
    Commit,
    Rollback,
}
