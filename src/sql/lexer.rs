use crate::common::error::{DbError, ErrorCode, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    StringLit(String),
    NumberLit(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Star,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Plus,
    Minus,
    Dot,
    Eof,
}

/// Single-pass UTF-8 SQL lexer.
pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::with_capacity(self.input.len() / 8 + 8);
        loop {
            let tok = self.next_token()?;
            let is_eof = tok == Token::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    #[inline]
    fn peek_char(&self) -> Option<char> {
        let bytes = self.input.as_bytes();
        if self.pos >= bytes.len() {
            return None;
        }
        let b = bytes[self.pos];
        if b.is_ascii() {
            Some(b as char)
        } else {
            self.input[self.pos..].chars().next()
        }
    }

    #[inline]
    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    #[inline]
    fn advance_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        let bytes = self.input.as_bytes();
        loop {
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            while let Some(ch) = self.peek_char() {
                if !ch.is_ascii() && ch.is_whitespace() {
                    self.advance_char();
                } else {
                    break;
                }
            }
            if self.rest().starts_with("--") {
                if let Some(nl_rel) = self.rest().find('\n') {
                    self.pos += nl_rel + 1;
                } else {
                    self.pos = self.input.len();
                }
                continue;
            }
            if self.rest().starts_with("/*") {
                self.pos += 2;
                let Some(end_rel) = self.rest().find("*/") else {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "unterminated block comment",
                    ));
                };
                self.pos += end_rel + 2;
                continue;
            }
            break;
        }
        Ok(())
    }

    fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace_and_comments()?;
        let Some(ch) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match ch {
            '(' => {
                self.advance_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.advance_char();
                Ok(Token::RParen)
            }
            '[' => {
                self.advance_char();
                Ok(Token::LBracket)
            }
            ']' => {
                self.advance_char();
                Ok(Token::RBracket)
            }
            ',' => {
                self.advance_char();
                Ok(Token::Comma)
            }
            ';' => {
                self.advance_char();
                Ok(Token::Semicolon)
            }
            '*' => {
                self.advance_char();
                Ok(Token::Star)
            }
            '+' => {
                self.advance_char();
                Ok(Token::Plus)
            }
            '-' => {
                self.advance_char();
                Ok(Token::Minus)
            }
            '.' => {
                self.advance_char();
                Ok(Token::Dot)
            }
            '=' => {
                self.advance_char();
                Ok(Token::Eq)
            }
            '!' => {
                self.advance_char();
                if self.peek_char() == Some('=') {
                    self.advance_char();
                    Ok(Token::Neq)
                } else {
                    Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "unexpected '!' (expected '!=')",
                    ))
                }
            }
            '<' => {
                self.advance_char();
                match self.peek_char() {
                    Some('=') => {
                        self.advance_char();
                        Ok(Token::Lte)
                    }
                    Some('>') => {
                        self.advance_char();
                        Ok(Token::Neq)
                    }
                    _ => Ok(Token::Lt),
                }
            }
            '>' => {
                self.advance_char();
                if self.peek_char() == Some('=') {
                    self.advance_char();
                    Ok(Token::Gte)
                } else {
                    Ok(Token::Gt)
                }
            }
            '\'' => self.read_string_literal(),
            '`' | '"' => self.read_quoted_ident(ch),
            c if c.is_ascii_digit() => self.read_number_literal(),
            c if is_ident_start(c) => Ok(self.read_ident()),
            other => Err(DbError::new(
                ErrorCode::SyntaxError,
                format!("unexpected character '{}'", other),
            )),
        }
    }

    fn read_string_literal(&mut self) -> Result<Token> {
        self.advance_char(); // opening quote
        let start = self.pos;
        let rest = &self.input[start..];
        let Some(first_quote) = rest.find('\'') else {
            return Err(DbError::new(
                ErrorCode::SyntaxError,
                "unterminated string literal",
            ));
        };
        let after_quote = start + first_quote + 1;
        if self.input.as_bytes().get(after_quote) != Some(&b'\'') {
            self.pos = after_quote;
            return Ok(Token::StringLit(rest[..first_quote].to_string()));
        }
        let mut out = String::with_capacity(first_quote + 16);
        out.push_str(&rest[..first_quote]);
        out.push('\'');
        self.pos = after_quote + 1;
        loop {
            let Some(ch) = self.advance_char() else {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    "unterminated string literal",
                ));
            };
            if ch == '\'' {
                if self.peek_char() == Some('\'') {
                    self.advance_char();
                    out.push('\'');
                } else {
                    return Ok(Token::StringLit(out));
                }
            } else {
                out.push(ch);
            }
        }
    }

    fn read_quoted_ident(&mut self, quote: char) -> Result<Token> {
        self.advance_char();
        let mut out = String::new();
        loop {
            let Some(ch) = self.advance_char() else {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    "unterminated quoted identifier",
                ));
            };
            if ch == quote {
                if self.peek_char() == Some(quote) {
                    self.advance_char();
                    out.push(quote);
                } else {
                    if out.is_empty() {
                        return Err(DbError::new(
                            ErrorCode::SyntaxError,
                            "empty quoted identifier",
                        ));
                    }
                    return Ok(Token::Ident(out));
                }
            } else {
                out.push(ch);
            }
        }
    }

    fn read_number_literal(&mut self) -> Result<Token> {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance_char();
            } else {
                break;
            }
        }
        if self.peek_char() == Some('.') {
            self.advance_char();
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() {
                    self.advance_char();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek_char(), Some('e' | 'E')) {
            self.advance_char();
            if matches!(self.peek_char(), Some('+' | '-')) {
                self.advance_char();
            }
            let mut has_exp_digit = false;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() {
                    has_exp_digit = true;
                    self.advance_char();
                } else {
                    break;
                }
            }
            if !has_exp_digit {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    "malformed exponent in numeric literal",
                ));
            }
        }
        Ok(Token::NumberLit(self.input[start..self.pos].to_string()))
    }

    fn read_ident(&mut self) -> Token {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if is_ident_continue(c) {
                self.advance_char();
            } else {
                break;
            }
        }
        Token::Ident(self.input[start..self.pos].to_string())
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
