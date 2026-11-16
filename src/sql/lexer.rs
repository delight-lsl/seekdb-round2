use crate::common::error::{DbError, ErrorCode, Result};
use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    Ident(Cow<'a, str>),
    StringLit(Cow<'a, str>),
    NumberLit(&'a str),
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

/// Single-pass UTF-8 SQL lexer borrowing slices from the input SQL string.
pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token<'a>>> {
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

    fn next_token(&mut self) -> Result<Token<'a>> {
        self.skip_whitespace_and_comments()?;
        let Some(ch) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match ch {
            '(' => {
                self.pos += 1;
                Ok(Token::LParen)
            }
            ')' => {
                self.pos += 1;
                Ok(Token::RParen)
            }
            '[' => {
                self.pos += 1;
                Ok(Token::LBracket)
            }
            ']' => {
                self.pos += 1;
                Ok(Token::RBracket)
            }
            ',' => {
                self.pos += 1;
                Ok(Token::Comma)
            }
            ';' => {
                self.pos += 1;
                Ok(Token::Semicolon)
            }
            '*' => {
                self.pos += 1;
                Ok(Token::Star)
            }
            '+' => {
                self.pos += 1;
                Ok(Token::Plus)
            }
            '-' => {
                self.pos += 1;
                Ok(Token::Minus)
            }
            '.' => {
                self.pos += 1;
                Ok(Token::Dot)
            }
            '=' => {
                self.pos += 1;
                Ok(Token::Eq)
            }
            '!' => {
                self.pos += 1;
                if self.peek_char() == Some('=') {
                    self.pos += 1;
                    Ok(Token::Neq)
                } else {
                    Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "unexpected '!' (expected '!=')",
                    ))
                }
            }
            '<' => {
                self.pos += 1;
                match self.peek_char() {
                    Some('=') => {
                        self.pos += 1;
                        Ok(Token::Lte)
                    }
                    Some('>') => {
                        self.pos += 1;
                        Ok(Token::Neq)
                    }
                    _ => Ok(Token::Lt),
                }
            }
            '>' => {
                self.pos += 1;
                if self.peek_char() == Some('=') {
                    self.pos += 1;
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

    fn read_string_literal(&mut self) -> Result<Token<'a>> {
        self.pos += 1; // opening quote
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
            return Ok(Token::StringLit(Cow::Borrowed(&rest[..first_quote])));
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
                    self.pos += 1;
                    out.push('\'');
                } else {
                    return Ok(Token::StringLit(Cow::Owned(out)));
                }
            } else {
                out.push(ch);
            }
        }
    }

    fn read_quoted_ident(&mut self, quote: char) -> Result<Token<'a>> {
        self.pos += 1;
        let start = self.pos;
        let rest = &self.input[start..];
        if let Some(first_quote) = rest.find(quote) {
            let after_quote = start + first_quote + 1;
            if self.input.as_bytes().get(after_quote) != Some(&(quote as u8)) {
                if first_quote == 0 {
                    return Err(DbError::new(
                        ErrorCode::SyntaxError,
                        "empty quoted identifier",
                    ));
                }
                self.pos = after_quote;
                return Ok(Token::Ident(Cow::Borrowed(&rest[..first_quote])));
            }
        }
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
                    self.pos += 1;
                    out.push(quote);
                } else {
                    if out.is_empty() {
                        return Err(DbError::new(
                            ErrorCode::SyntaxError,
                            "empty quoted identifier",
                        ));
                    }
                    return Ok(Token::Ident(Cow::Owned(out)));
                }
            } else {
                out.push(ch);
            }
        }
    }

    fn read_number_literal(&mut self) -> Result<Token<'a>> {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < bytes.len() && bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < bytes.len() && matches!(bytes[self.pos], b'e' | b'E') {
            self.pos += 1;
            if self.pos < bytes.len() && matches!(bytes[self.pos], b'+' | b'-') {
                self.pos += 1;
            }
            let mut has_exp_digit = false;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
                has_exp_digit = true;
                self.pos += 1;
            }
            if !has_exp_digit {
                return Err(DbError::new(
                    ErrorCode::SyntaxError,
                    "malformed exponent in numeric literal",
                ));
            }
        }
        Ok(Token::NumberLit(&self.input[start..self.pos]))
    }

    fn read_ident(&mut self) -> Token<'a> {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() && is_ident_continue_byte(bytes[self.pos]) {
            self.pos += 1;
        }
        Token::Ident(Cow::Borrowed(&self.input[start..self.pos]))
    }
}

#[inline]
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

#[inline]
fn is_ident_continue_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
