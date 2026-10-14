mod greedy;
mod hypothesis;
mod joint;
mod search;

use crate::tokenization::TokenId;

pub use greedy::StandardTdtGreedyExecutor;
pub use hypothesis::{TdtHypothesis, TdtTokenAlignment};
pub use joint::{TdtJointLayout, TdtJointOutput};
pub use search::TdtSearch;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtDecodingConfig {
    num_token_classes: usize,
    blank_token_id: TokenId,
    durations: Vec<usize>,
    max_symbols_per_step: usize,
}

impl TdtDecodingConfig {
    pub fn new(
        num_token_classes: usize,
        blank_token_id: TokenId,
        durations: impl Into<Vec<usize>>,
        max_symbols_per_step: usize,
    ) -> Self {
        Self {
            num_token_classes,
            blank_token_id,
            durations: durations.into(),
            max_symbols_per_step,
        }
    }

    pub fn num_token_classes(&self) -> usize {
        self.num_token_classes
    }

    pub fn blank_token_id(&self) -> TokenId {
        self.blank_token_id
    }

    pub fn durations(&self) -> &[usize] {
        &self.durations
    }

    pub fn max_symbols_per_step(&self) -> usize {
        self.max_symbols_per_step
    }
}
