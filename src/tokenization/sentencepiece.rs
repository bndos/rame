use std::path::Path;

use sentencepiece::SentencePieceProcessor;

use super::{Decoder, TokenId, TokenizationError};
use crate::RameResult;

/// Decoder backed by a SentencePiece `.model` file.
pub struct SentencePieceDecoder {
    processor: SentencePieceProcessor,
}

impl SentencePieceDecoder {
    pub fn from_file(path: impl AsRef<Path>) -> RameResult<Self> {
        let path = path.as_ref();
        let processor =
            SentencePieceProcessor::open(path).map_err(|error| TokenizationError::Load {
                path: path.to_owned(),
                message: error.to_string(),
            })?;
        Ok(Self { processor })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.processor.len()
    }

    pub fn piece_id(&self, piece: &str) -> RameResult<Option<TokenId>> {
        self.processor.piece_to_id(piece).map_err(|error| {
            TokenizationError::Decode {
                message: error.to_string(),
            }
            .into()
        })
    }

    pub fn unknown_id(&self) -> TokenId {
        self.processor.unk_id()
    }
}

impl Decoder for SentencePieceDecoder {
    fn decode(&self, token_ids: &[TokenId], skip_special_tokens: bool) -> RameResult<String> {
        let filtered = skip_special_tokens.then(|| {
            token_ids
                .iter()
                .copied()
                .filter(|id| {
                    Some(*id) != self.processor.bos_id()
                        && Some(*id) != self.processor.eos_id()
                        && Some(*id) != self.processor.pad_id()
                })
                .collect::<Vec<_>>()
        });
        let ids = filtered.as_deref().unwrap_or(token_ids);
        self.processor.decode_piece_ids(ids).map_err(|error| {
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
        token_ids
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }
}
