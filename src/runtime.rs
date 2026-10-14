pub use rame_core::runtime::{
    DecodeBatch, Decoder, ModelLoader, ModelRunner, ProcessedBatch, Processor, RuntimeError,
    StandardModelRunner, TransducerDecoding, TransducerEncoder, TransducerEncoding,
    TransducerModelRunner,
};

pub mod transducer {
    pub use rame_core::runtime::transducer::{
        JointNetwork, MaskedPredictionNetwork, PredictionNetwork, TransducerDecoding,
        TransducerEncoder, TransducerEncoding, TransducerModelRunner,
    };

    pub mod tdt {
        pub use rame_core::runtime::transducer::tdt::*;

        #[cfg(all(feature = "mlx", target_os = "macos"))]
        pub use crate::backends::mlx::tdt::MlxTdtGreedyExecutor;
    }
}

pub use transducer::{JointNetwork, MaskedPredictionNetwork, PredictionNetwork};
