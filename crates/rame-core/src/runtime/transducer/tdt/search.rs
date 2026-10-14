use crate::RameResult;
use crate::runtime::{JointNetwork, PredictionNetwork};

use super::TdtHypothesis;

pub trait TdtSearch<P, J>
where
    P: PredictionNetwork,
    J: JointNetwork,
{
    type Encoding;

    fn decode_many(
        &mut self,
        encoding: Self::Encoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>>;

    fn decode(
        &mut self,
        encoding: Self::Encoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<TdtHypothesis> {
        crate::runtime::expect_one(
            self.decode_many(encoding, predictor, joint)?,
            "TDT search hypothesis",
        )
    }
}
