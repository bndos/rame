use crate::layout::LayoutLabel;

/// Maps class ids using the official `PP-DocLayout_plus-L` `label_list` order.
///
/// Source: <https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L/blob/main/config.json>
pub(super) fn label_for_class_id(class_id: i64) -> LayoutLabel {
    match class_id {
        0 => LayoutLabel::Title,             // paragraph_title
        1 => LayoutLabel::Image,             // image
        2 => LayoutLabel::Text,              // text
        3 => LayoutLabel::PageNumber,        // number
        4 => LayoutLabel::Abstract,          // abstract
        5 => LayoutLabel::Content,           // content
        6 => LayoutLabel::FigureCaption,     // figure_title
        7 => LayoutLabel::Formula,           // formula
        8 => LayoutLabel::Table,             // table
        9 => LayoutLabel::Reference,         // reference
        10 => LayoutLabel::Title,            // doc_title
        11 => LayoutLabel::Footnote,         // footnote
        12 => LayoutLabel::Header,           // header
        13 => LayoutLabel::Algorithm,        // algorithm
        14 => LayoutLabel::Footer,           // footer
        15 => LayoutLabel::Seal,             // seal
        16 => LayoutLabel::Chart,            // chart
        17 => LayoutLabel::FormulaNumber,    // formula_number
        18 => LayoutLabel::AsideText,        // aside_text
        19 => LayoutLabel::ReferenceContent, // reference_content
        other => LayoutLabel::Unknown(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::layout::LayoutLabel;
    use crate::models::pp_doclayout::plus::labels::label_for_class_id;

    #[test]
    fn maps_pp_doclayout_plus_class_ids_to_layout_labels() {
        assert_eq!(label_for_class_id(0), LayoutLabel::Title);
        assert_eq!(label_for_class_id(1), LayoutLabel::Image);
        assert_eq!(label_for_class_id(2), LayoutLabel::Text);
        assert_eq!(label_for_class_id(3), LayoutLabel::PageNumber);
        assert_eq!(label_for_class_id(4), LayoutLabel::Abstract);
        assert_eq!(label_for_class_id(5), LayoutLabel::Content);
        assert_eq!(label_for_class_id(6), LayoutLabel::FigureCaption);
        assert_eq!(label_for_class_id(7), LayoutLabel::Formula);
        assert_eq!(label_for_class_id(8), LayoutLabel::Table);
        assert_eq!(label_for_class_id(9), LayoutLabel::Reference);
        assert_eq!(label_for_class_id(10), LayoutLabel::Title);
        assert_eq!(label_for_class_id(11), LayoutLabel::Footnote);
        assert_eq!(label_for_class_id(12), LayoutLabel::Header);
        assert_eq!(label_for_class_id(13), LayoutLabel::Algorithm);
        assert_eq!(label_for_class_id(14), LayoutLabel::Footer);
        assert_eq!(label_for_class_id(15), LayoutLabel::Seal);
        assert_eq!(label_for_class_id(16), LayoutLabel::Chart);
        assert_eq!(label_for_class_id(17), LayoutLabel::FormulaNumber);
        assert_eq!(label_for_class_id(18), LayoutLabel::AsideText);
        assert_eq!(label_for_class_id(19), LayoutLabel::ReferenceContent);
    }

    #[test]
    fn preserves_unknown_pp_doclayout_plus_class_ids() {
        assert_eq!(
            label_for_class_id(99),
            LayoutLabel::Unknown("99".to_string())
        );
    }
}
