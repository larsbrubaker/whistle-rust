//! The page-wide drop target: wraps the demo's root so a WAV dropped anywhere on the page
//! reaches [`DemoHost::transcribe_file`](crate::DemoHost::transcribe_file). Drops bubble from
//! the widget under the cursor to the root, so the root sees every drop no child took.
//! Native drops (paths) are read from disk; browser drops carry the bytes.

use std::sync::Arc;

use agg_gui::{DrawCtx, Event, EventResult, Rect, Size, Widget};

/// What a drop calls with the file's name and bytes.
type OnDrop = Box<dyn FnMut(&str, Arc<[u8]>)>;

/// Wraps one child; hands every dropped file to `on_drop` (name, bytes).
pub struct DropTarget {
    bounds: Rect,
    children: Vec<Box<dyn Widget>>,
    on_drop: OnDrop,
}

impl DropTarget {
    pub fn new(child: Box<dyn Widget>, on_drop: impl FnMut(&str, Arc<[u8]>) + 'static) -> Self {
        DropTarget {
            bounds: Rect::default(),
            children: vec![child],
            on_drop: Box::new(on_drop),
        }
    }
}

impl Widget for DropTarget {
    fn type_name(&self) -> &'static str {
        "DropTarget"
    }
    fn bounds(&self) -> Rect {
        self.bounds
    }
    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut Vec<Box<dyn Widget>> {
        &mut self.children
    }

    fn layout(&mut self, available: Size) -> Size {
        if let Some(child) = self.children.first_mut() {
            child.layout(available);
            child.set_bounds(Rect::new(0.0, 0.0, available.width, available.height));
        }
        self.bounds = Rect::new(0.0, 0.0, available.width, available.height);
        available
    }

    fn paint(&mut self, _ctx: &mut dyn DrawCtx) {}

    fn on_event(&mut self, event: &Event) -> EventResult {
        match event {
            Event::FileDataDropped { files, .. } => {
                if let Some(file) = files.first() {
                    (self.on_drop)(&file.name, Arc::clone(&file.bytes));
                }
                EventResult::Consumed
            }
            Event::FileDropped { paths, .. } => {
                if let Some(path) = paths.first() {
                    if let Ok(bytes) = std::fs::read(path) {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        (self.on_drop)(&name, bytes.into());
                    }
                }
                EventResult::Consumed
            }
            _ => EventResult::Ignored,
        }
    }
}
