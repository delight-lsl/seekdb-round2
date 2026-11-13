use crate::catalog::schema::ColumnSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::DataType;
use crate::sql::ast::{
    BinaryOp, Expr, LiteralValue, OrderByItem, OrderByTarget, QueryVector, SelectItem, Statement,
};
use crate::sql::lexer::{Lexer, Token};

/// Recursive-descent SQL parser producing `Statement` AST nodes.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn parse_sql(sql: &str) -> Result<Statement> {
        let mut lexer = Lexer::new(sql);
        let tokens = lexer.tokenize()?;
        let mut parser = Self { tokens, pos: 0 };
        parser.parse_statement()
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn peek_next(&self) -> &Token {
        self.tokens.get(self.pos + 1).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token {
        let idx = self.pos;
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        &self.tokens[idx]
    }

    fn check_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Token::Ident(s) if s.eq_ignore_ascii_case(kw))
    }

    fn consume_kw(&mut self, kw: &str) -> bool {
        if self.check_kw(kw) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        if self.consume_kw(kw) {
            Ok(())
        } else {
            Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected keyword '{}', found {:?}", kw, self.peek()),
            ))
        }
    }

    fn expect_token(&mut self, expected: &Token) -> Result<()> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected {:?}, found {:?}", expected, self.peek()),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.advance().clone() {
            Token::Ident(s) => Ok(s),
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected identifier, found {:?}", other),
            )),
        }
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        if *self.peek() == Token::Eof {
            return Err(DbError::new(ErrorCode::SyntaxError, "empty SQL statement"));
        }

        let stmt = match self.peek() {
            Token::Ident(kw) => match kw.to_ascii_uppercase().as_str() {
                "CREATE" => self.parse_create()?,
                "INSERT" => self.parse_insert()?,
                "SELECT" => self.parse_select()?,
                "UPDATE" => self.parse_update()?,
                "DELETE" => self.parse_delete()?,
                "BEGIN" => self.parse_begin()?,
                "COMMIT" => self.parse_commit()?,
                "ROLLBACK" => self.parse_rollback()?,
                "ALTER" | "DROP" | "TRUNCATE" | "REPLACE" | "SHOW" | "EXPLAIN" | "PRAGMA"
                | "USE" | "GRANT" | "REVOKE" | "SET" | "WITH" => {
                    return Err(DbError::new(
                        ErrorCode::UnsupportedSql,
                        format!("unsupported SQL statement: {}", kw),
                    ));
                }
                other => {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        format!("unexpected SQL keyword '{}'", other),
                    ));
                }
            },
            other => {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    format!("unexpected token at start of SQL: {:?}", other),
                ));
            }
        };

        // Allow optional trailing semicolons
        while *self.peek() == Token::Semicolon {
            self.advance();
        }
        if *self.peek() != Token::Eof {
            return Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("unexpected trailing token: {:?}", self.peek()),
            ));
        }
        Ok(stmt)
    }

    fn parse_begin(&mut self) -> Result<Statement> {
        self.expect_kw("BEGIN")?;
        let _ = self.consume_kw("TRANSACTION");
        Ok(Statement::Begin)
    }

    fn parse_commit(&mut self) -> Result<Statement> {
        self.expect_kw("COMMIT")?;
        let _ = self.consume_kw("TRANSACTION");
        Ok(Statement::Commit)
    }

    fn parse_rollback(&mut self) -> Result<Statement> {
        self.expect_kw("ROLLBACK")?;
        let _ = self.consume_kw("TRANSACTION");
        Ok(Statement::Rollback)
    }

    fn parse_create(&mut self) -> Result<Statement> {
        self.expect_kw("CREATE")?;
        if self.consume_kw("TABLE") {
            let name = self.expect_ident()?;
            self.expect_token(&Token::LParen)?;
            let mut columns: Vec<ColumnSchema> = Vec::new();
            let mut table_pk_col: Option<String> = None;
            loop {
                if *self.peek() == Token::RParen {
                    break;
                }
                if self.consume_kw("PRIMARY") {
                    self.expect_kw("KEY")?;
                    self.expect_token(&Token::LParen)?;
                    let pk_col = self.expect_ident()?;
                    self.expect_token(&Token::RParen)?;
                    if table_pk_col.is_some() {
                        return Err(DbError::new(
                            ErrorCode::SyntaxError,
                            "multiple primary keys are not supported",
                        ));
                    }
                    table_pk_col = Some(pk_col);
                } else {
                    let col_name = self.expect_ident()?;
                    let data_type = self.parse_data_type()?;
                    let mut is_primary_key = false;
                    loop {
                        if self.consume_kw("PRIMARY") {
                            self.expect_kw("KEY")?;
                            is_primary_key = true;
                        } else if self.consume_kw("NOT") {
                            self.expect_kw("NULL")?;
                        } else if self.consume_kw("NULL") {
                            // nullable allowed
                        } else {
                            break;
                        }
                    }
                    columns.push(ColumnSchema {
                        name: col_name,
                        data_type,
                        is_primary_key,
                    });
                }
                if *self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            if let Some(pk_name) = table_pk_col {
                if columns.iter().any(|c| c.is_primary_key) {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "multiple primary keys are not supported",
                    ));
                }
                let mut found = false;
                for col in &mut columns {
                    if col.name.eq_ignore_ascii_case(&pk_name) {
                        col.is_primary_key = true;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(DbError::new(
                        ErrorCode::ColumnNotFound,
                        format!("PRIMARY KEY column '{}' not found", pk_name),
                    ));
                }
            }
            Ok(Statement::CreateTable { name, columns })
        } else if self.consume_kw("VECTOR") {
            self.expect_kw("INDEX")?;
            let index_name = self.expect_ident()?;
            self.expect_kw("ON")?;
            let table_name = self.expect_ident()?;
            self.expect_token(&Token::LParen)?;
            let column_name = self.expect_ident()?;
            self.expect_token(&Token::RParen)?;
            Ok(Statement::CreateVectorIndex {
                index_name,
                table_name,
                column_name,
            })
        } else {
            Err(DbError::new(
                ErrorCode::UnsupportedSql,
                format!("unsupported CREATE statement near {:?}", self.peek()),
            ))
        }
    }

    fn parse_data_type(&mut self) -> Result<DataType> {
        let type_kw = self.expect_ident()?;
        match type_kw.to_ascii_uppercase().as_str() {
            "BIGINT" => Ok(DataType::BigInt),
            "VARCHAR" => {
                if *self.peek() == Token::LParen {
                    self.advance();
                    let len = self.parse_usize_lit()?;
                    self.expect_token(&Token::RParen)?;
                    Ok(DataType::Varchar(len))
                } else {
                    Ok(DataType::Varchar(65535))
                }
            }
            "VECTOR" => {
                self.expect_token(&Token::LParen)?;
                let dim = self.parse_usize_lit()?;
                self.expect_token(&Token::RParen)?;
                if dim == 0 {
                    return Err(DbError::new(
                        ErrorCode::InvalidVector,
                        "VECTOR dimension must be > 0",
                    ));
                }
                Ok(DataType::Vector(dim))
            }
            other => Err(DbError::new(
                ErrorCode::UnsupportedSql,
                format!("unsupported column data type '{}'", other),
            )),
        }
    }

    fn parse_usize_lit(&mut self) -> Result<usize> {
        match self.advance().clone() {
            Token::NumberLit(s) => s.parse::<usize>().map_err(|_| {
                DbError::new(
                    ErrorCode::SyntaxError,
                    format!("expected positive integer, found '{}'", s),
                )
            }),
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected integer literal, found {:?}", other),
            )),
        }
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_kw("INSERT")?;
        self.expect_kw("INTO")?;
        let table_name = self.expect_ident()?;
        let columns = if *self.peek() == Token::LParen {
            self.advance();
            let mut cols = Vec::new();
            loop {
                cols.push(self.expect_ident()?);
                if *self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            Some(cols)
        } else {
            None
        };
        self.expect_kw("VALUES")?;
        let mut rows = Vec::new();
        loop {
            self.expect_token(&Token::LParen)?;
            let mut vals = Vec::new();
            loop {
                vals.push(self.parse_literal()?);
                if *self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            rows.push(vals);
            if *self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(Statement::Insert {
            table_name,
            columns,
            rows,
        })
    }

    /// Take the string literal at the cursor out of the token stream (no copy).
    fn take_string_lit(&mut self) -> Option<String> {
        if !matches!(self.peek(), Token::StringLit(_)) {
            return None;
        }
        let idx = self.pos;
        self.advance();
        match std::mem::replace(&mut self.tokens[idx], Token::Comma) {
            Token::StringLit(s) => Some(s),
            _ => None,
        }
    }

    fn parse_literal(&mut self) -> Result<LiteralValue> {
        if let Some(s) = self.take_string_lit() {
            return Ok(LiteralValue::Str(s));
        }
        match self.peek().clone() {
            Token::Ident(s) if s.eq_ignore_ascii_case("NULL") => {
                self.advance();
                Ok(LiteralValue::Null)
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(LiteralValue::Str(s))
            }
            Token::LBracket => {
                self.advance();
                let mut vec = Vec::new();
                if *self.peek() != Token::RBracket {
                    loop {
                        let f = self.parse_signed_f64()?;
                        vec.push(f);
                        if *self.peek() == Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect_token(&Token::RBracket)?;
                Ok(LiteralValue::Vector(vec))
            }
            Token::Minus | Token::Plus | Token::NumberLit(_) => self.parse_signed_number(),
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected literal value, found {:?}", other),
            )),
        }
    }

    fn parse_signed_number(&mut self) -> Result<LiteralValue> {
        let mut sign = "";
        if *self.peek() == Token::Minus {
            self.advance();
            sign = "-";
        } else if *self.peek() == Token::Plus {
            self.advance();
        }
        match self.advance().clone() {
            Token::NumberLit(s) => {
                let full = if sign.is_empty() {
                    s
                } else {
                    format!("{}{}", sign, s)
                };
                if !full.contains('.') && !full.contains('e') && !full.contains('E') {
                    if let Ok(iv) = full.parse::<i64>() {
                        return Ok(LiteralValue::Int(iv));
                    }
                }
                let fv: f64 = full.parse().map_err(|_| {
                    DbError::new(
                        ErrorCode::SyntaxError,
                        format!("invalid numeric literal '{}'", full),
                    )
                })?;
                Ok(LiteralValue::Float(fv))
            }
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("expected number literal, found {:?}", other),
            )),
        }
    }

    fn parse_signed_f64(&mut self) -> Result<f64> {
        let mut sign = "";
        if *self.peek() == Token::Minus {
            self.advance();
            sign = "-";
        } else if *self.peek() == Token::Plus {
            self.advance();
        }
        match self.advance().clone() {
            Token::NumberLit(s) => {
                let full = if sign.is_empty() {
                    s
                } else {
                    format!("{}{}", sign, s)
                };
                full.parse::<f64>().map_err(|_| {
                    DbError::new(
                        ErrorCode::InvalidVector,
                        format!("invalid float literal '{}'", full),
                    )
                })
            }
            Token::Ident(s) => {
                // Allow catching NaN/Infinity tokens cleanly as InvalidVector
                let full = if sign.is_empty() {
                    s
                } else {
                    format!("{}{}", sign, s)
                };
                if let Ok(v) = full.parse::<f64>() {
                    Ok(v)
                } else {
                    Err(DbError::new(
                        ErrorCode::InvalidVector,
                        format!("invalid element '{}' in vector", full),
                    ))
                }
            }
            other => Err(DbError::new(
                ErrorCode::InvalidVector,
                format!("expected number in vector, found {:?}", other),
            )),
        }
    }

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect_kw("SELECT")?;
        if self.check_kw("DISTINCT") {
            return Err(DbError::new(
                ErrorCode::UnsupportedSql,
                "DISTINCT is not supported",
            ));
        }

        let mut items = Vec::new();
        loop {
            items.push(self.parse_select_item()?);
            if *self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        self.expect_kw("FROM")?;
        let table_name = self.expect_ident()?;

        if self.check_kw("JOIN")
            || self.check_kw("INNER")
            || self.check_kw("LEFT")
            || self.check_kw("RIGHT")
            || self.check_kw("CROSS")
        {
            return Err(DbError::new(
                ErrorCode::UnsupportedSql,
                "JOIN is not supported",
            ));
        }

        let where_clause = if self.consume_kw("WHERE") {
            Some(self.parse_expr()?)
        } else {
            None
        };

        if self.check_kw("GROUP") || self.check_kw("HAVING") || self.check_kw("UNION") {
            return Err(DbError::new(
                ErrorCode::UnsupportedSql,
                format!("unsupported clause near {:?}", self.peek()),
            ));
        }

        let mut order_by = Vec::new();
        if self.consume_kw("ORDER") {
            self.expect_kw("BY")?;
            loop {
                let target = if self.check_kw("L2_DISTANCE")
                    && *self.peek_next() == Token::LParen
                {
                    let (col, query) = self.parse_l2_distance_call()?;
                    OrderByTarget::L2Distance { column: col, query }
                } else {
                    OrderByTarget::ColumnOrAlias(self.expect_ident()?)
                };
                let asc = if self.consume_kw("ASC") {
                    true
                } else if self.consume_kw("DESC") {
                    false
                } else {
                    true
                };
                order_by.push(OrderByItem { target, asc });
                if *self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let limit = if self.consume_kw("LIMIT") {
            Some(self.parse_usize_lit()?)
        } else {
            None
        };

        if self.check_kw("OFFSET") {
            return Err(DbError::new(
                ErrorCode::UnsupportedSql,
                "OFFSET is not supported",
            ));
        }

        Ok(Statement::Select {
            items,
            table_name,
            where_clause,
            order_by,
            limit,
        })
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        if *self.peek() == Token::Star {
            self.advance();
            return Ok(SelectItem::Wildcard);
        }
        if let Token::Ident(fn_name) = self.peek() {
            if matches!(
                fn_name.to_ascii_uppercase().as_str(),
                "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
            ) && *self.peek_next() == Token::LParen
            {
                return Err(DbError::new(
                    ErrorCode::UnsupportedSql,
                    format!("aggregate function {} is not supported", fn_name),
                ));
            }
        }
        if self.check_kw("L2_DISTANCE") && *self.peek_next() == Token::LParen {
            let (column, query) = self.parse_l2_distance_call()?;
            let alias = if self.consume_kw("AS") {
                Some(self.expect_ident()?)
            } else if let Token::Ident(next_kw) = self.peek() {
                if !next_kw.eq_ignore_ascii_case("FROM") {
                    Some(self.expect_ident()?)
                } else {
                    None
                }
            } else {
                None
            };
            return Ok(SelectItem::L2Distance {
                column,
                query,
                alias,
            });
        }

        let name = self.expect_ident()?;
        let alias = if self.consume_kw("AS") {
            Some(self.expect_ident()?)
        } else {
            None
        };
        Ok(SelectItem::Column { name, alias })
    }

    fn parse_l2_distance_call(&mut self) -> Result<(String, QueryVector)> {
        self.expect_kw("L2_DISTANCE")?;
        self.expect_token(&Token::LParen)?;
        // Support both L2_DISTANCE(col, '[...]') and L2_DISTANCE('[...]', col)
        let (col, query) = match self.peek() {
            Token::StringLit(_) | Token::LBracket => {
                let q = self.parse_query_vector()?;
                self.expect_token(&Token::Comma)?;
                let c = self.expect_ident()?;
                (c, q)
            }
            Token::Ident(_) => {
                let c = self.expect_ident()?;
                self.expect_token(&Token::Comma)?;
                let q = self.parse_query_vector()?;
                (c, q)
            }
            other => {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    format!("unexpected argument in L2_DISTANCE: {:?}", other),
                ));
            }
        };
        self.expect_token(&Token::RParen)?;
        Ok((col, query))
    }

    fn parse_query_vector(&mut self) -> Result<QueryVector> {
        if let Some(s) = self.take_string_lit() {
            return Ok(QueryVector::Text(s));
        }
        match self.peek().clone() {
            Token::StringLit(s) => {
                self.advance();
                Ok(QueryVector::Text(s))
            }
            Token::LBracket => {
                let lit = self.parse_literal()?;
                match lit {
                    LiteralValue::Vector(v) => Ok(QueryVector::Numbers(v)),
                    _ => unreachable!(),
                }
            }
            other => Err(DbError::new(
                ErrorCode::InvalidVector,
                format!("expected vector argument, found {:?}", other),
            )),
        }
    }

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect_kw("UPDATE")?;
        let table_name = self.expect_ident()?;
        self.expect_kw("SET")?;
        let mut assignments = Vec::new();
        loop {
            let col = self.expect_ident()?;
            self.expect_token(&Token::Eq)?;
            let val = self.parse_literal()?;
            assignments.push((col, val));
            if *self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        let where_clause = if self.consume_kw("WHERE") {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::Update {
            table_name,
            assignments,
            where_clause,
        })
    }

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect_kw("DELETE")?;
        self.expect_kw("FROM")?;
        let table_name = self.expect_ident()?;
        let where_clause = if self.consume_kw("WHERE") {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::Delete {
            table_name,
            where_clause,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while self.consume_kw("OR") {
            let right = self.parse_and()?;
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_not()?;
        while self.consume_kw("AND") {
            let right = self.parse_not()?;
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if self.consume_kw("NOT") {
            let inner = self.parse_not()?;
            return Ok(Expr::UnaryNot(Box::new(inner)));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let left = self.parse_primary()?;
        if self.consume_kw("IS") {
            let negated = self.consume_kw("NOT");
            self.expect_kw("NULL")?;
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
            });
        }
        let negated_pred = self.consume_kw("NOT");
        if self.consume_kw("BETWEEN") {
            let low = self.parse_primary()?;
            self.expect_kw("AND")?;
            let high = self.parse_primary()?;
            let between_expr = Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(left.clone()),
                    op: BinaryOp::Gte,
                    right: Box::new(low),
                }),
                op: BinaryOp::And,
                right: Box::new(Expr::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Lte,
                    right: Box::new(high),
                }),
            };
            return Ok(if negated_pred {
                Expr::UnaryNot(Box::new(between_expr))
            } else {
                between_expr
            });
        }
        if self.consume_kw("IN") {
            self.expect_token(&Token::LParen)?;
            let mut items = Vec::new();
            loop {
                items.push(self.parse_primary()?);
                if *self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect_token(&Token::RParen)?;
            let mut iter = items.into_iter();
            let first = iter.next().ok_or_else(|| {
                DbError::new(ErrorCode::SyntaxError, "IN (...) requires at least one value")
            })?;
            let mut or_expr = Expr::Binary {
                left: Box::new(left.clone()),
                op: BinaryOp::Eq,
                right: Box::new(first),
            };
            for item in iter {
                or_expr = Expr::Binary {
                    left: Box::new(or_expr),
                    op: BinaryOp::Or,
                    right: Box::new(Expr::Binary {
                        left: Box::new(left.clone()),
                        op: BinaryOp::Eq,
                        right: Box::new(item),
                    }),
                };
            }
            return Ok(if negated_pred {
                Expr::UnaryNot(Box::new(or_expr))
            } else {
                or_expr
            });
        }
        if negated_pred {
            return Err(DbError::new(
                ErrorCode::SyntaxError,
                "unexpected NOT in comparison expression",
            ));
        }
        let op = match self.peek() {
            Token::Eq => Some(BinaryOp::Eq),
            Token::Neq => Some(BinaryOp::Neq),
            Token::Lt => Some(BinaryOp::Lt),
            Token::Lte => Some(BinaryOp::Lte),
            Token::Gt => Some(BinaryOp::Gt),
            Token::Gte => Some(BinaryOp::Gte),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            let right = self.parse_primary()?;
            Ok(Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::LParen => {
                self.advance();
                if self.check_kw("SELECT") {
                    return Err(DbError::new(
                        ErrorCode::UnsupportedSql,
                        "subqueries are not supported",
                    ));
                }
                let inner = self.parse_expr()?;
                self.expect_token(&Token::RParen)?;
                Ok(inner)
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("NULL") => {
                self.advance();
                Ok(Expr::Literal(LiteralValue::Null))
            }
            Token::Ident(s)
                if s.eq_ignore_ascii_case("L2_DISTANCE") && *self.peek_next() == Token::LParen =>
            {
                let (column, query) = self.parse_l2_distance_call()?;
                Ok(Expr::L2Distance { column, query })
            }
            Token::Ident(s) => {
                self.advance();
                Ok(Expr::Column(s))
            }
            Token::StringLit(_)
            | Token::LBracket
            | Token::Minus
            | Token::Plus
            | Token::NumberLit(_) => Ok(Expr::Literal(self.parse_literal()?)),
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("unexpected token in expression: {:?}", other),
            )),
        }
    }
}
