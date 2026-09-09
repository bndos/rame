use crate::RameResult;

use super::TokenId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeResult {
    Complete(String),
    /// Text ending in an incomplete byte sequence that needs more token IDs.
    Partial(String),
}

impl DecodeResult {
    pub fn from_decoded(text: String) -> Self {
        if text.ends_with('\u{FFFD}') {
            Self::Partial(text)
        } else {
            Self::Complete(text)
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Complete(text) | Self::Partial(text) => text,
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }

    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial(_))
    }

    pub fn into_string(self) -> String {
        match self {
            Self::Complete(text) | Self::Partial(text) => text,
        }
    }
}

impl From<DecodeResult> for String {
    fn from(result: DecodeResult) -> Self {
        result.into_string()
    }
}

impl From<String> for DecodeResult {
    fn from(text: String) -> Self {
        Self::from_decoded(text)
    }
}

pub trait Decoder: Send + Sync {
    fn decode(&self, token_ids: &[TokenId], skip_special_tokens: bool) -> RameResult<DecodeResult>;
}

#[cfg(test)]
mod tests {
    use super::DecodeResult;

    #[test]
    fn identifies_incomplete_trailing_utf8() {
        assert!(DecodeResult::from_decoded("hello".into()).is_complete());
        assert!(DecodeResult::from_decoded("hello\u{FFFD}".into()).is_partial());
    }
}
