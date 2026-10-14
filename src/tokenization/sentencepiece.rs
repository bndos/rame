use std::path::Path;

use sentencepiece_rust::SentencePieceProcessor;

use super::{Decoder, TokenId, TokenizationError};
use crate::RameResult;

/// Decoder backed by a SentencePiece `.model` file.
pub struct SentencePieceDecoder {
    processor: SentencePieceProcessor,
    unknown_id: TokenId,
}

impl SentencePieceDecoder {
    pub fn from_file(path: impl AsRef<Path>) -> RameResult<Self> {
        let path = path.as_ref();
        let processor =
            SentencePieceProcessor::open(path).map_err(|error| TokenizationError::Load {
                path: path.to_owned(),
                message: error.to_string(),
            })?;
        let unknown_id =
            TokenId::try_from(processor.vocab().unk_id).map_err(|_| TokenizationError::Load {
                path: path.to_owned(),
                message: "SentencePiece model has no unknown token".into(),
            })?;
        Ok(Self {
            processor,
            unknown_id,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.processor.piece_size()
    }

    pub fn piece_id(&self, piece: &str) -> RameResult<Option<TokenId>> {
        Ok(self
            .processor
            .vocab()
            .piece_to_id(piece.as_bytes())
            .and_then(|id| TokenId::try_from(id).ok()))
    }

    pub fn unknown_id(&self) -> TokenId {
        self.unknown_id
    }
}

impl Decoder for SentencePieceDecoder {
    fn decode(&self, token_ids: &[TokenId], skip_special_tokens: bool) -> RameResult<String> {
        let filtered = skip_special_tokens.then(|| {
            let vocab = self.processor.vocab();
            token_ids
                .iter()
                .copied()
                .filter(|id| {
                    Some(*id) != TokenId::try_from(vocab.bos_id).ok()
                        && Some(*id) != TokenId::try_from(vocab.eos_id).ok()
                        && Some(*id) != TokenId::try_from(vocab.pad_id).ok()
                })
                .collect::<Vec<_>>()
        });
        let ids = filtered.as_deref().unwrap_or(token_ids);
        let ids = ids
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| TokenizationError::Decode {
                message: error.to_string(),
            })?;
        self.processor.decode(&ids).map_err(|error| {
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{Decoder, SentencePieceDecoder};

    #[test]
    fn decodes_sentencepiece_ids_and_skips_special_tokens() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sentencepiece.model");
        let decoder = SentencePieceDecoder::from_file(path).unwrap();

        assert_eq!(decoder.vocabulary_size(), 24);
        assert_eq!(decoder.unknown_id(), 0);
        assert_eq!(decoder.piece_id("▁hello").unwrap(), Some(4));
        assert_eq!(decoder.piece_id("missing").unwrap(), None);
        assert_eq!(decoder.decode(&[4, 6], false).unwrap(), "hello world");
        assert_eq!(
            decoder.decode(&[1, 4, 6, 2, 3], true).unwrap(),
            "hello world"
        );
    }
}
