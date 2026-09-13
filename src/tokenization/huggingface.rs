use std::path::Path;

use crate::RameResult;

use super::{Decoder, TokenId, TokenizationError};

/// Decoder backed by a Hugging Face `tokenizer.json`.
#[derive(Clone)]
pub struct HuggingFaceDecoder {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceDecoder {
    pub fn from_file(path: impl AsRef<Path>) -> RameResult<Self> {
        let path = path.as_ref();
        let tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|error| TokenizationError::Load {
                path: path.to_owned(),
                message: error.to_string(),
            })?;
        Ok(Self { tokenizer })
    }
}

impl From<tokenizers::Tokenizer> for HuggingFaceDecoder {
    fn from(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl Decoder for HuggingFaceDecoder {
    fn decode(&self, token_ids: &[TokenId], skip_special_tokens: bool) -> RameResult<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                TokenizationError::Decode {
                    message: error.to_string(),
                }
                .into()
            })
    }

    fn decode_many(
        &self,
        token_ids: &[&[TokenId]],
        skip_special_tokens: bool,
    ) -> RameResult<Vec<String>> {
        self.tokenizer
            .decode_batch(token_ids, skip_special_tokens)
            .map_err(|error| {
                TokenizationError::Decode {
                    message: error.to_string(),
                }
                .into()
            })
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    #[test]
    fn decodes_with_the_wrapped_tokenizer() {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<unk>".to_owned(), 0),
                    ("hello".to_owned(), 1),
                    ("world".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let decoder = HuggingFaceDecoder::from(Tokenizer::new(model));

        assert_eq!(decoder.decode(&[1, 2], true).unwrap(), "hello world",);
        assert_eq!(
            decoder.decode_many(&[&[1], &[2, 1]], true).unwrap(),
            ["hello", "world hello"],
        );
    }
}
