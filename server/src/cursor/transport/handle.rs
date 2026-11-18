//! Provides the request-scoped input, subscription, and terminal interface.

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{connect, proto::agent::v1 as pb},
        services::observability::CursorTraceRecorder,
    },
    Error, Result,
};

use super::{OutputHub, TransportAdmission, TransportLifecycle};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportParent {
    pub request_id: String,
    pub tool_call_id: String,
}

#[derive(Clone)]
pub struct TransportHandle {
    request_id: String,
    commands: mpsc::Sender<TransportCommand>,
    output: Arc<OutputHub>,
    conversation_id: Arc<OnceLock<String>>,
    parent: Arc<OnceLock<TransportParent>>,
    /// Cursor >= 3.30 sends the client-generated invocation id via an
    /// `invoke_model` frame inside BidiAppend. The RunInferenceSSE bridge must
    /// echo this exact id back so the client matches the response queue.
    invocation_id: Arc<OnceLock<String>>,
    trace: CursorTraceRecorder,
    lifecycle: TransportLifecycle,
    disconnect: CancellationToken,
}

impl TransportHandle {
    pub(crate) fn new(
        request_id: String,
        commands: mpsc::Sender<TransportCommand>,
        output: Arc<OutputHub>,
        trace: CursorTraceRecorder,
    ) -> Self {
        Self {
            request_id,
            commands,
            output,
            conversation_id: Arc::new(OnceLock::new()),
            parent: Arc::new(OnceLock::new()),
            invocation_id: Arc::new(OnceLock::new()),
            trace,
            lifecycle: TransportLifecycle::new(),
            disconnect: CancellationToken::new(),
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn set_conversation_id(&self, conversation_id: &str) -> Result<()> {
        if conversation_id.is_empty() {
            return Err(Error::Protocol("Cursor conversation id is required".into()));
        }
        if self
            .conversation_id
            .get()
            .is_some_and(|current| current != conversation_id)
        {
            return Err(Error::Protocol(format!(
                "conflicting conversation ids for request {}",
                self.request_id
            )));
        }
        let _ = self.conversation_id.set(conversation_id.into());
        Ok(())
    }

    pub fn conversation_id(&self) -> Option<&str> {
        self.conversation_id.get().map(String::as_str)
    }

    /// Store the client-generated invocation id (first write wins). The id is
    /// delivered via an `invoke_model` frame on BidiAppend after the SSE stream
    /// opens, so a plain `OnceLock` gives us the "wait until available" slot.
    pub fn set_invocation_id(&self, invocation_id: &str) {
        if invocation_id.is_empty() {
            return;
        }
        let _ = self.invocation_id.set(invocation_id.to_string());
    }

    pub fn invocation_id(&self) -> Option<&str> {
        self.invocation_id.get().map(String::as_str)
    }

    pub fn set_parent(&self, parent: TransportParent) -> Result<()> {
        if parent.request_id.is_empty() || parent.tool_call_id.is_empty() {
            return Err(Error::Protocol("Cursor parent ids are required".into()));
        }
        if self.parent.get().is_some_and(|current| current != &parent) {
            return Err(Error::Protocol(format!(
                "conflicting parent ids for request {}",
                self.request_id
            )));
        }
        let _ = self.parent.set(parent);
        Ok(())
    }

    pub fn parent(&self) -> Option<&TransportParent> {
        self.parent.get()
    }

    pub async fn command(&self, command: TransportCommand) -> Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| Error::RunNotFound(self.request_id.clone()))
    }

    pub async fn disconnect(&self) {
        let _ = self.commands.send(TransportCommand::Disconnect).await;
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<Bytes> {
        self.output.subscribe()
    }

    pub fn emit_frame(&self, frame: Bytes) -> bool {
        self.output.emit(frame)
    }

    pub fn emit(&self, message: &pb::AgentServerMessage) -> Result<()> {
        // PROBE: record what the run engine writes into this request's hub, so
        // we can compare against what the SSE bridge reads from the same hub.
        let kind = match message.message.as_ref() {
            None => "none",
            Some(pb::agent_server_message::Message::InteractionUpdate(u)) => {
                match u.message.as_ref() {
                    Some(pb::interaction_update::Message::TextDelta(_)) => "text_delta",
                    Some(pb::interaction_update::Message::ThinkingDelta(_)) => "thinking_delta",
                    Some(pb::interaction_update::Message::TokenDelta(_)) => "token_delta",
                    Some(pb::interaction_update::Message::Heartbeat(_)) => "heartbeat",
                    Some(pb::interaction_update::Message::TurnEnded(_)) => "turn_ended",
                    Some(_) => "interaction_other",
                    None => "interaction_none",
                }
            }
            Some(pb::agent_server_message::Message::ExecServerMessage(_)) => "exec",
            Some(pb::agent_server_message::Message::ExecServerControlMessage(_)) => "exec_control",
            Some(pb::agent_server_message::Message::ConversationCheckpointUpdate(_)) => {
                "checkpoint_update"
            }
            Some(pb::agent_server_message::Message::KvServerMessage(_)) => "kv",
            Some(pb::agent_server_message::Message::InteractionQuery(_)) => "interaction_query",
        };
        tracing::info!(
            request_id = %self.request_id,
            kind,
            "PROBE hub emit"
        );
        if self.emit_frame(connect::encode_message(message)?) {
            Ok(())
        } else {
            Err(Error::RunNotFound(self.request_id.clone()))
        }
    }

    pub(crate) fn close_output(&self) -> bool {
        self.output.close()
    }

    pub(crate) fn trace(&self) -> Option<&CursorTraceRecorder> {
        Some(&self.trace)
    }

    pub(crate) fn accepting_appends(&self) -> bool {
        self.lifecycle.is_open()
    }

    pub(crate) fn admit(&self) -> Result<TransportAdmission> {
        self.lifecycle
            .admit()
            .ok_or_else(|| Error::RunNotFound(self.request_id.clone()))
    }

    pub(crate) fn begin_close(&self) {
        self.lifecycle.begin_close();
    }

    pub(crate) fn admissions_drained(&self) -> bool {
        self.lifecycle.admissions_drained()
    }

    pub(crate) async fn wait_admissions_drained(&self) {
        self.lifecycle.wait_admissions_drained().await;
    }

    pub(crate) fn mark_draining(&self) {
        self.lifecycle.mark_draining();
    }

    pub(crate) fn reopen(&self) {
        self.lifecycle.reopen();
    }

    pub(crate) fn close_transport(&self) {
        self.lifecycle.close();
    }

    pub(crate) async fn wait_transport_closed(&self) {
        self.lifecycle.wait_closed().await;
    }

    pub(crate) fn disconnect_token(&self) -> CancellationToken {
        self.disconnect.clone()
    }

    pub(crate) fn mark_disconnected(&self) {
        self.disconnect.cancel();
    }
}
