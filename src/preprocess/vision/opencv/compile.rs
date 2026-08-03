use crate::preprocess::vision::{TensorLayout, VisionOp};

pub(super) fn compile(ops: &mut Vec<VisionOp>) {
    fuse_normalize_permute(ops);
}

fn fuse_normalize_permute(ops: &mut Vec<VisionOp>) {
    let mut index = 0;

    while index + 1 < ops.len() {
        match (ops[index], ops[index + 1]) {
            (VisionOp::NormalizeImage(normalize), VisionOp::Permute(permute))
                if permute.layout == TensorLayout::Nchw =>
            {
                ops[index] = VisionOp::NormalizeAndPermute { normalize, permute };
                ops.remove(index + 1);
            }
            _ => index += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::preprocess::vision::opencv::compile::compile;
    use crate::preprocess::vision::{Interpolation, NormalizeImage, Permute, Resize, VisionOp};

    #[test]
    fn fuses_adjacent_normalize_and_nchw_permute() {
        let normalize = NormalizeImage::scale(NormalizeImage::INV_255);
        let permute = Permute::nchw();
        let mut ops = vec![
            VisionOp::NormalizeImage(normalize),
            VisionOp::Permute(permute),
        ];

        compile(&mut ops);

        assert_eq!(
            ops,
            vec![VisionOp::NormalizeAndPermute { normalize, permute }]
        );
    }

    #[test]
    fn preserves_non_adjacent_normalize_and_permute() {
        let normalize = NormalizeImage::scale(NormalizeImage::INV_255);
        let permute = Permute::nchw();
        let resize = Resize::fixed_square(800, Interpolation::Cubic);
        let mut ops = vec![
            VisionOp::NormalizeImage(normalize),
            VisionOp::Resize(resize),
            VisionOp::Permute(permute),
        ];

        compile(&mut ops);

        assert_eq!(
            ops,
            vec![
                VisionOp::NormalizeImage(normalize),
                VisionOp::Resize(resize),
                VisionOp::Permute(permute),
            ]
        );
    }

    #[test]
    fn keeps_scanning_after_fusion() {
        let normalize = NormalizeImage::scale(NormalizeImage::INV_255);
        let permute = Permute::nchw();
        let mut ops = vec![
            VisionOp::NormalizeImage(normalize),
            VisionOp::Permute(permute),
            VisionOp::NormalizeImage(normalize),
            VisionOp::Permute(permute),
        ];

        compile(&mut ops);

        assert_eq!(
            ops,
            vec![
                VisionOp::NormalizeAndPermute { normalize, permute },
                VisionOp::NormalizeAndPermute { normalize, permute },
            ]
        );
    }
}
