use crate::RameResult;

pub type TokenId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    token_ids: Vec<TokenId>,
}

impl Encoding {
    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }

    pub fn into_token_ids(self) -> Vec<TokenId> {
        self.token_ids
    }
}

impl From<Vec<TokenId>> for Encoding {
    fn from(token_ids: Vec<TokenId>) -> Self {
        Self { token_ids }
    }
}

pub trait Encoder: Send + Sync {
    fn encode(&self, input: &str) -> RameResult<Encoding>;

    fn encode_many(&self, inputs: &[&str]) -> RameResult<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::RameResult;

    use super::{Encoder, Encoding, TokenId};

    struct LengthEncoder;

    impl Encoder for LengthEncoder {
        fn encode(&self, input: &str) -> RameResult<Encoding> {
            Ok(vec![input.len() as TokenId].into())
        }
    }

    #[test]
    fn encodes_many_with_the_default_serial_implementation() {
        let encoded = LengthEncoder.encode_many(&["one", "three"]).unwrap();

        assert_eq!(
            encoded.iter().map(Encoding::token_ids).collect::<Vec<_>>(),
            vec![&[3][..], &[5][..]],
        );
    }
}
