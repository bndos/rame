use crate::preprocess::vision::VisionOp;

pub(super) fn compile(_ops: &mut Vec<VisionOp>) {}

#[cfg(test)]
mod tests {
    use crate::preprocess::vision::opencv::compile::compile;
    use crate::preprocess::vision::{Interpolation, NormalizeImage, Resize, ToTensor, VisionOp};

    #[test]
    fn leaves_tensor_conversion_ops_unchanged() {
        let tensor = ToTensor::nchw().normalize(NormalizeImage::scale(NormalizeImage::INV_255));
        let mut ops = vec![
            VisionOp::Resize(Resize::fixed_square(800, Interpolation::Cubic)),
            VisionOp::ToTensor(tensor),
        ];

        compile(&mut ops);

        assert_eq!(
            ops,
            vec![
                VisionOp::Resize(Resize::fixed_square(800, Interpolation::Cubic)),
                VisionOp::ToTensor(tensor),
            ],
        );
    }
}
