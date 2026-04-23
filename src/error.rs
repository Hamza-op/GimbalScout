use std::path::PathBuf;

use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Message(String),

    #[error("missing dependency: {what}. Provide {hint}.")]
    MissingDependency {
        what: &'static str,
        hint: &'static str,
    },

    #[error("failed to run `{cmd}`: {source}")]
    CommandFailed {
        cmd: String,
        #[source]
        source: std::io::Error,
    },

    #[error("command `{cmd}` exited with code {code}")]
    CommandNonZero { cmd: String, code: i32 },

    #[error("failed to parse {what}: {source}")]
    ParseFailed {
        what: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("unsupported input: {0}")]
    Unsupported(String),

    #[error("analysis cancelled")]
    Cancelled,

    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl AppError {
    pub fn exit_code(&self) -> i32 {
        match self {
            AppError::MissingDependency { .. } => 2,
            AppError::Cancelled => 3,
            _ => 1,
        }
    }
}
