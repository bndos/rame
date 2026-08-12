use crate::layout::LayoutLabel;

/// Maps class ids using the official `PP-DocLayoutV3` `label_list` order.
///
/// Source: <https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/blob/main/inference.yml>
pub(super) fn label_for_class_id(class_id: i64) -> LayoutLabel {
    match class_id {
        0 => LayoutLabel::Abstract,          // abstract
        1 => LayoutLabel::Algorithm,         // algorithm
        2 => LayoutLabel::AsideText,         // aside_text
        3 => LayoutLabel::Chart,             // chart
        4 => LayoutLabel::Content,           // content
        5 => LayoutLabel::Formula,           // display_formula
        6 => LayoutLabel::Title,             // doc_title
        7 => LayoutLabel::FigureCaption,     // figure_title
        8 => LayoutLabel::Footer,            // footer
        9 => LayoutLabel::FooterImage,       // footer_image
        10 => LayoutLabel::Footnote,         // footnote
        11 => LayoutLabel::FormulaNumber,    // formula_number
        12 => LayoutLabel::Header,           // header
        13 => LayoutLabel::HeaderImage,      // header_image
        14 => LayoutLabel::Image,            // image
        15 => LayoutLabel::Formula,          // inline_formula
        16 => LayoutLabel::PageNumber,       // number
        17 => LayoutLabel::Title,            // paragraph_title
        18 => LayoutLabel::Reference,        // reference
        19 => LayoutLabel::ReferenceContent, // reference_content
        20 => LayoutLabel::Seal,             // seal
        21 => LayoutLabel::Table,            // table
        22 => LayoutLabel::Text,             // text
        23 => LayoutLabel::Text,             // vertical_text
        24 => LayoutLabel::Footnote,         // vision_footnote
        other => LayoutLabel::Unknown(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::layout::LayoutLabel;
    use crate::models::pp_doclayout::v3::labels::label_for_class_id;

    #[test]
    fn maps_pp_doclayout_v3_class_ids_to_layout_labels() {
        assert_eq!(label_for_class_id(0), LayoutLabel::Abstract);
        assert_eq!(label_for_class_id(1), LayoutLabel::Algorithm);
        assert_eq!(label_for_class_id(2), LayoutLabel::AsideText);
        assert_eq!(label_for_class_id(3), LayoutLabel::Chart);
        assert_eq!(label_for_class_id(4), LayoutLabel::Content);
        assert_eq!(label_for_class_id(5), LayoutLabel::Formula);
        assert_eq!(label_for_class_id(6), LayoutLabel::Title);
        assert_eq!(label_for_class_id(7), LayoutLabel::FigureCaption);
        assert_eq!(label_for_class_id(8), LayoutLabel::Footer);
        assert_eq!(label_for_class_id(9), LayoutLabel::FooterImage);
        assert_eq!(label_for_class_id(10), LayoutLabel::Footnote);
        assert_eq!(label_for_class_id(11), LayoutLabel::FormulaNumber);
        assert_eq!(label_for_class_id(12), LayoutLabel::Header);
        assert_eq!(label_for_class_id(13), LayoutLabel::HeaderImage);
        assert_eq!(label_for_class_id(14), LayoutLabel::Image);
        assert_eq!(label_for_class_id(15), LayoutLabel::Formula);
        assert_eq!(label_for_class_id(16), LayoutLabel::PageNumber);
        assert_eq!(label_for_class_id(17), LayoutLabel::Title);
        assert_eq!(label_for_class_id(18), LayoutLabel::Reference);
        assert_eq!(label_for_class_id(19), LayoutLabel::ReferenceContent);
        assert_eq!(label_for_class_id(20), LayoutLabel::Seal);
        assert_eq!(label_for_class_id(21), LayoutLabel::Table);
        assert_eq!(label_for_class_id(22), LayoutLabel::Text);
        assert_eq!(label_for_class_id(23), LayoutLabel::Text);
        assert_eq!(label_for_class_id(24), LayoutLabel::Footnote);
    }

    #[test]
    fn preserves_unknown_pp_doclayout_v3_class_ids() {
        assert_eq!(
            label_for_class_id(99),
            LayoutLabel::Unknown("99".to_string())
        );
    }
}
