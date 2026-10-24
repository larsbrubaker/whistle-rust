//! Layout tests for the transcript (`ui.rs`): its text lies inside the scroll view's content,
//! and a long transcript scrolls with the newest words in view. Also pins the agg-gui
//! limitation the transcript works around (a top-anchored column behind a `Rebuilder` inside
//! a `ScrollView` is laid out off screen): when that test starts failing, agg-gui has fixed it.

use std::sync::Arc;

use agg_gui::{FlexColumn, Label, Rebuilder, Rect, ScrollView, Size, Widget};

use super::transcript;
use crate::install_fonts;
use crate::state::{DemoState, Phase};

/// The bounds of the widget with id `id`, in the coordinates of `root`'s content (parent
/// offsets summed; Y up).
fn find(root: &dyn Widget, id: &str, origin: (f64, f64)) -> Option<Rect> {
    for child in root.children() {
        let b = child.bounds();
        let at = (origin.0 + b.x, origin.1 + b.y);
        if child.id() == Some(id) {
            return Some(Rect::new(at.0, at.1, b.width, b.height));
        }
        if let Some(found) = find(child.as_ref(), id, at) {
            return Some(found);
        }
    }
    None
}

fn scroll_view(widget: &dyn Widget) -> &ScrollView {
    widget
        .as_any()
        .and_then(|a| a.downcast_ref::<ScrollView>())
        .expect("the transcript is a ScrollView")
}

#[test]
fn rebuilder_hides_a_top_anchored_column_from_scroll_view_relayout() {
    // The minimal agg-gui repro: ScrollView > Rebuilder > FlexColumn(top-anchored) > Label.
    // The ScrollView measures its content at an unbounded height and re-lays it out at the
    // measured height only when a direct child of its content lies outside; the column is the
    // Rebuilder's child, not the ScrollView content's, so the label stays near 1e308.
    let font = install_fonts(1.0);
    let mut view = ScrollView::new(Box::new(Rebuilder::new(
        || 0,
        move || {
            Box::new(FlexColumn::new().with_top_anchor(true).add(Box::new(
                Label::new("hello", Arc::clone(&font)).with_name("l"),
            ))) as Box<dyn Widget>
        },
    )));
    view.layout(Size::new(400.0, 300.0));
    let label = find(&view, "l", (0.0, 0.0)).expect("label laid out");
    assert!(
        label.y > 1e300,
        "agg-gui now lays this out on screen (label at y = {}): the transcript's workaround \
         (an un-anchored column, ui.rs) can go",
        label.y
    );
}

#[test]
fn the_transcript_text_lies_inside_the_scroll_content() {
    let font = install_fonts(1.0);
    let state = DemoState::new();
    state.set_phase(Phase::Ready);
    state.set_final("Please export the bracket.", None, Vec::new(), 100.0, 3.0);
    let mut view = transcript(&state, &font);
    view.layout(Size::new(600.0, 300.0));
    let label = find(view.as_ref(), "committed", (0.0, 0.0)).expect("committed text shown");
    assert!(
        label.y >= 0.0 && label.y + label.height <= 300.0,
        "label at {label:?}"
    );
}

#[test]
fn a_long_transcript_scrolls_and_stays_at_the_newest_words() {
    let font = install_fonts(1.0);
    let state = DemoState::new();
    state.set_phase(Phase::Ready);
    state.start_listening();
    let mut view = transcript(&state, &font);
    view.layout(Size::new(400.0, 200.0));
    for i in 0..80 {
        state.apply_update(
            &format!("sentence number {i} goes here."),
            false,
            "",
            None,
            1.0,
            1.0,
        );
        view.layout(Size::new(400.0, 200.0));
    }
    let scroll = scroll_view(view.as_ref());
    assert!(
        scroll.max_scroll_value() > 200.0,
        "the transcript overflows the view"
    );
    // Stuck to the bottom: scrolled all the way, so the newest words are in view.
    assert!(
        (scroll.scroll_offset() - scroll.max_scroll_value()).abs() < 1.0,
        "offset {} of {}",
        scroll.scroll_offset(),
        scroll.max_scroll_value()
    );
}
