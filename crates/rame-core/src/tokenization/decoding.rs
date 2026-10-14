use crate::RameResult;

use super::TokenId;

pub trait Decoder: Send + Sync {
    fn decode(&self, token_ids: &[TokenId], skip_special_tokens: bool) -> RameResult<String>;

    fn decode_many(
        &self,
        token_ids: &[&[TokenId]],
        skip_special_tokens: bool,
    ) -> RameResult<Vec<String>>;
}
