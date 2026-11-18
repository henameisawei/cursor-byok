//! Wires the Cursor-facing API routes.

mod agent_ws;
pub mod bidi;
mod handlers;
pub mod proxy;
mod run_inference;
mod run_sse;

pub use handlers::router;
