use crate::RameResult;
use crate::runtime::{JointNetwork, PredictionNetwork, TransducerEncoding};

use super::{TdtHypothesis, TdtJointOutput};

pub trait TdtSearch<P, J>
where
    P: PredictionNetwork,
    J: JointNetwork<Output = TdtJointOutput>,
{
    fn decode_many(
        &mut self,
        encoding: TransducerEncoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>>;

    fn decode(
        &mut self,
        encoding: TransducerEncoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<TdtHypothesis> {
        crate::runtime::expect_one(
            self.decode_many(encoding, predictor, joint)?,
            "TDT search hypothesis",
        )
    }
}
