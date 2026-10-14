use crate::{RameError, RameResult};

pub(crate) fn expect_one<T>(results: Vec<T>, stage: &'static str) -> RameResult<T> {
    let [result]: [_; 1] =
        results
            .try_into()
            .map_err(|results: Vec<T>| RameError::InvalidBatchLength {
                stage,
                expected: 1,
                actual: results.len(),
            })?;

    Ok(result)
}
