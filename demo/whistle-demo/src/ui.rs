//! The demo's widget tree: title, the controls (Record/Stop, transcribe a file), the options
//! row, the status line, the transcript and the license note, all in one page-wide drop target
//! (`drop_target.rs`). Parts that change with [`DemoState`] sit in `Rebuilder`s keyed on its
//! version; the options row is built once, so typing in the keywords field keeps its focus.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use agg_gui::text::Font;
use agg_gui::{
    current_visuals, Button, Checkbox, ComboBox, FlexColumn, FlexRow, HAnchor, Hyperlink, Insets,
    Label, Rebuilder, ScrollView, SegmentedControl, TextField, VAnchor, Widget,
};

use crate::drop_target::DropTarget;
use crate::state::{DemoState, BEAM_CHOICES, LANGUAGE_CHOICES};
use crate::{DemoHost, APP_TITLE, MODEL_LICENSE_URL, SOURCE_URL};

/// Widget id of the transcript's scroll view.
pub const TRANSCRIPT_ID: &str = "transcript";

pub fn build_root(state: DemoState, host: Rc<dyn DemoHost>, font: Arc<Font>) -> Box<dyn Widget> {
    let column = FlexColumn::new()
        .with_panel_bg()
        .with_top_anchor(true)
        .with_gap(12.0)
        .with_inner_padding(Insets::all(20.0))
        .add(Box::new(
            Label::new(APP_TITLE, Arc::clone(&font))
                .with_font_size(22.0)
                .with_strong(true),
        ))
        .add(Box::new(
            Label::new(
                "Live speech-to-text with Whistle, running entirely in this page \
                 (pure Rust, WebAssembly, one thread). Nothing you say leaves your device.",
                Arc::clone(&font),
            )
            .with_wrap(true)
            .with_dim(true),
        ))
        .add(controls(&state, &host, &font))
        .add(options_row(&state, &font))
        .add(status(&state, &font))
        .add_flex(transcript(&state, &font), 1.0)
        .add(license(&host, &font));
    let drop_state = state.clone();
    Box::new(DropTarget::new(Box::new(column), move |name, bytes| {
        if drop_state.can_start() {
            host_drop(&host, &drop_state, name, bytes);
        }
    }))
}

fn host_drop(host: &Rc<dyn DemoHost>, state: &DemoState, name: &str, bytes: Arc<[u8]>) {
    host.transcribe_file(name, bytes, &state.options());
}

fn controls(state: &DemoState, host: &Rc<dyn DemoHost>, font: &Arc<Font>) -> Box<dyn Widget> {
    let version_state = state.clone();
    let state = state.clone();
    let host = Rc::clone(host);
    let font = Arc::clone(font);
    Box::new(
        Rebuilder::new(
            move || version_state.version(),
            move || {
                let listening = state.is_listening();
                let label = if listening { "Stop" } else { "Record" };
                let (record_state, record_host) = (state.clone(), Rc::clone(&host));
                let enabled_state = state.clone();
                let record = Button::new(label, Arc::clone(&font))
                    .with_font_size(16.0)
                    .with_enabled_fn(move || {
                        enabled_state.is_listening() || enabled_state.can_start()
                    })
                    .on_click(move || {
                        if record_state.is_listening() {
                            record_host.stop_recording();
                        } else if record_state.can_start() {
                            record_host.start_recording(&record_state.options());
                        }
                    })
                    .with_name("record");
                let (file_state, file_host) = (state.clone(), Rc::clone(&host));
                let enabled_state = state.clone();
                let pick = Button::new("Transcribe a WAV file\u{2026}", Arc::clone(&font))
                    .with_enabled_fn(move || enabled_state.can_start())
                    .on_click(move || {
                        if file_state.can_start() {
                            file_host.pick_file(&file_state.options());
                        }
                    })
                    .with_name("pick-file");
                Box::new(
                    FlexRow::new()
                        .with_gap(10.0)
                        .add(Box::new(record))
                        .add(Box::new(pick))
                        .add(Box::new(
                            Label::new("or drop one on the page", Arc::clone(&font))
                                .with_dim(true)
                                .with_v_anchor(VAnchor::CENTER),
                        )),
                ) as Box<dyn Widget>
            },
        )
        .with_h_anchor(HAnchor::STRETCH),
    )
}

fn options_row(state: &DemoState, font: &Arc<Font>) -> Box<dyn Widget> {
    let options = state.options();
    let lang_state = state.clone();
    let language = ComboBox::new(
        LANGUAGE_CHOICES.to_vec(),
        options.language,
        Arc::clone(font),
    )
    .with_fit_width(true)
    .on_change(move |i| lang_state.edit_options(|o| o.language = i));
    let beams_cell = Rc::new(Cell::new(options.beams));
    let beams_state = state.clone();
    let beams = SegmentedControl::new(
        BEAM_CHOICES
            .iter()
            .map(|b| format!("{b} beam{}", if *b == 1 { "" } else { "s" }))
            .collect(),
        beams_cell,
        Arc::clone(font),
    )
    .on_change(move |i| beams_state.edit_options(|o| o.beams = i));
    let keywords_state = state.clone();
    let keywords = TextField::new(Arc::clone(font))
        .with_placeholder("keywords, comma separated")
        .with_text(options.keywords)
        .on_change(move |text| keywords_state.edit_options(|o| o.keywords = text.to_string()));
    let stamps_state = state.clone();
    let stamps = Checkbox::new("word timestamps", Arc::clone(font), options.word_timestamps)
        .on_change(move |on| stamps_state.edit_options(|o| o.word_timestamps = on));
    Box::new(
        FlexRow::new()
            .with_gap(10.0)
            .with_h_anchor(HAnchor::STRETCH)
            .add(Box::new(
                Label::new("Language", Arc::clone(font)).with_v_anchor(VAnchor::CENTER),
            ))
            .add(Box::new(language))
            .add(Box::new(beams))
            .add_flex(Box::new(keywords), 1.0)
            .add(Box::new(stamps)),
    )
}

fn status(state: &DemoState, font: &Arc<Font>) -> Box<dyn Widget> {
    let version_state = state.clone();
    let state = state.clone();
    let font = Arc::clone(font);
    Box::new(
        Rebuilder::new(
            move || version_state.version(),
            move || {
                Box::new(
                    Label::new(state.status_line(), Arc::clone(&font))
                        .with_strong(true)
                        .with_name("status"),
                ) as Box<dyn Widget>
            },
        )
        .with_h_anchor(HAnchor::STRETCH),
    )
}

pub(crate) fn transcript(state: &DemoState, font: &Arc<Font>) -> Box<dyn Widget> {
    let version_state = state.clone();
    let state = state.clone();
    let font = Arc::clone(font);
    let body = Rebuilder::new(
        move || version_state.version(),
        move || {
            let committed = state.committed();
            let pending = state.pending();
            // Not top-anchored: a ScrollView measures its content at an unbounded height and
            // re-lays it out at the measured height only when one of the content's *direct*
            // children lies outside it. Behind the Rebuilder the column is not a direct child,
            // so a top-anchored column would leave its labels near 1e308, off screen
            // (`ui_tests.rs` pins both cases).
            let mut column = FlexColumn::new()
                .with_gap(6.0)
                .with_h_anchor(HAnchor::STRETCH);
            if committed.is_empty() && pending.is_empty() {
                column = column.add(Box::new(
                    Label::new(
                        "Press Record and speak (English, German, French, Spanish, Italian, \
                         Dutch or Polish), or transcribe a WAV file.",
                        Arc::clone(&font),
                    )
                    .with_wrap(true)
                    .with_dim(true),
                ));
            }
            if !committed.is_empty() {
                column = column.add(Box::new(
                    Label::new(committed, Arc::clone(&font))
                        .with_font_size(18.0)
                        .with_wrap(true)
                        .with_h_anchor(HAnchor::STRETCH)
                        .with_name("committed"),
                ));
            }
            if !pending.is_empty() {
                column = column.add(Box::new(
                    Label::new(pending, Arc::clone(&font))
                        .with_font_size(18.0)
                        .with_wrap(true)
                        .with_dim(true)
                        .with_h_anchor(HAnchor::STRETCH)
                        .with_name("pending"),
                ));
            }
            let words = state.words_line();
            if !words.is_empty() {
                column = column.add(Box::new(
                    Label::new(words, Arc::clone(&font))
                        .with_font_size(12.0)
                        .with_wrap(true)
                        .with_dim(true)
                        .with_h_anchor(HAnchor::STRETCH)
                        .with_name("words"),
                ));
            }
            Box::new(column) as Box<dyn Widget>
        },
    )
    .with_h_anchor(HAnchor::STRETCH);
    // Stick to the bottom: while streaming, the newest words stay in view unless the user
    // scrolls up to read back. The fade matches the panel the view sits on.
    Box::new(
        ScrollView::new(Box::new(body))
            .with_stick_to_bottom(true)
            .with_fade_color(current_visuals().panel_fill)
            .with_name(TRANSCRIPT_ID),
    )
}

fn license(host: &Rc<dyn DemoHost>, font: &Arc<Font>) -> Box<dyn Widget> {
    let (model_host, source_host) = (Rc::clone(host), Rc::clone(host));
    Box::new(
        FlexRow::new()
            .with_gap(6.0)
            .add(Box::new(
                Label::new(
                    "Model: Whistle by Cactus Compute, Inc., Apache-2.0.",
                    Arc::clone(font),
                )
                .with_font_size(12.0)
                .with_dim(true)
                .with_v_anchor(VAnchor::CENTER),
            ))
            .add(Box::new(
                Hyperlink::new("Model card and license", Arc::clone(font))
                    .with_font_size(12.0)
                    .on_click(move || model_host.open_url(MODEL_LICENSE_URL)),
            ))
            .add(Box::new(
                Hyperlink::new("Source (Apache-2.0)", Arc::clone(font))
                    .with_font_size(12.0)
                    .on_click(move || source_host.open_url(SOURCE_URL)),
            )),
    )
}

#[cfg(test)]
#[path = "ui_tests.rs"]
mod tests;
