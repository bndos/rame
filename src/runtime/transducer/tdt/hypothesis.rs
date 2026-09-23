use crate::tokenization::TokenId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtTokenAlignment {
    token_id: TokenId,
    start_frame: usize,
    duration: usize,
}

impl TdtTokenAlignment {
    pub fn token_id(self) -> TokenId {
        self.token_id
    }

    pub fn start_frame(self) -> usize {
        self.start_frame
    }

    pub fn duration(self) -> usize {
        self.duration
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtHypothesis {
    tokens: Vec<TdtTokenAlignment>,
}

impl TdtHypothesis {
    pub(super) fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub(super) fn push(&mut self, token_id: TokenId, start_frame: usize, duration: usize) {
        self.tokens.push(TdtTokenAlignment {
            token_id,
            start_frame,
            duration,
        });
    }

    pub fn tokens(&self) -> &[TdtTokenAlignment] {
        &self.tokens
    }
}
