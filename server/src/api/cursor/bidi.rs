//! Accepts ordered Cursor Bidi append requests and routes them by request_id.
use prost::Message;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{
            events,
            proto::{agent::v1 as agent, aiserver::v1 as ai},
        },
        transport::{TransportParent, TransportRegistry},
    },
    Error, Result,
};

pub struct DecodedAppend {
    pub request_id: String,
    pub seqno: i64,
    pub message: agent::AgentClientMessage,
    /// Client-generated invocation id extracted from a 3.30
    /// `RunInferenceClientMessage.invoke_model` frame. Stored on the transport
    /// handle so the RunInferenceSSE bridge can echo it back verbatim.
    pub invocation_id: Option<String>,
    /// True when the frame was a new-protocol inference control frame (e.g.
    /// `invoke_model`). Such frames only carry the invocation id / prompt and
    /// must NOT be replayed into the legacy run engine.
    pub is_inference_control: bool,
}

impl DecodedAppend {
    pub fn model_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str())
            .filter(|model| !model.is_empty())
            .or_else(|| {
                request
                    .model_details
                    .as_ref()
                    .map(|model| model.model_id.as_str())
                    .filter(|model| !model.is_empty())
            })
    }

    pub fn conversation_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request.conversation_id.as_deref()
    }

    pub fn is_background_task_completion(&self) -> bool {
        let Some(agent::agent_client_message::Message::RunRequest(request)) =
            self.message.message.as_ref()
        else {
            return false;
        };
        matches!(
            request
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(agent::conversation_action::Action::BackgroundTaskCompletionAction(_))
        )
    }

    pub fn trace_metadata(&self) -> serde_json::Value {
        let Some(message) = self.message.message.as_ref() else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": "empty",
            });
        };
        let agent::agent_client_message::Message::RunRequest(request) = message else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": client_message_type(message),
            });
        };
        let (action_type, history_messages, history_images) = request
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref())
            .map(|action| match action {
                agent::conversation_action::Action::UserMessageAction(action) => {
                    let history = action.conversation_history.as_ref();
                    (
                        "user_message",
                        history.map_or(0, |history| history.messages.len()),
                        history.map_or(0, history_image_count),
                    )
                }
                agent::conversation_action::Action::BackgroundTaskCompletionAction(_) => {
                    ("background_task_completion", 0, 0)
                }
                agent::conversation_action::Action::ExecutePlanAction(_) => ("execute_plan", 0, 0),
                agent::conversation_action::Action::SummarizeAction(_) => ("summarize", 0, 0),
                _ => ("other", 0, 0),
            })
            .unwrap_or(("none", 0, 0));
        let state = request.conversation_state.as_ref();
        serde_json::json!({
            "append_seqno": self.seqno,
            "message_type": "run_request",
            "conversation_id": request.conversation_id,
            "model_id": self.model_id(),
            "action_type": action_type,
            "conversation_history_messages": history_messages,
            "conversation_history_images": history_images,
            "root_message_count": state.map_or(0, |state| state.root_prompt_messages_json.len()),
            "turn_count": state.map_or(0, |state| state.turns.len()),
        })
    }
}

fn client_message_type(message: &agent::agent_client_message::Message) -> &'static str {
    use agent::agent_client_message::Message;
    match message {
        Message::RunRequest(_) => "run_request",
        Message::ExecClientMessage(_) => "exec_client_message",
        Message::ExecClientControlMessage(_) => "exec_client_control_message",
        Message::KvClientMessage(_) => "kv_client_message",
        Message::ConversationAction(_) => "conversation_action",
        Message::InteractionResponse(_) => "interaction_response",
        Message::ClientHeartbeat(_) => "client_heartbeat",
        Message::PrewarmRequest(_) => "prewarm_request",
    }
}

fn history_image_count(history: &agent::ConversationHistory) -> usize {
    use agent::{
        conversation_history_message::Message,
        conversation_history_tool_result_content::Content as ToolContent,
        conversation_history_user_content::Content as UserContent,
    };
    history
        .messages
        .iter()
        .map(|message| match message.message.as_ref() {
            Some(Message::User(user)) => user
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(UserContent::Image(_))))
                .count(),
            Some(Message::Tool(tool)) => tool
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(ToolContent::Image(_))))
                .count(),
            _ => 0,
        })
        .sum()
}

pub fn decode(request: &ai::BidiAppendRequest) -> Result<DecodedAppend> {
    let request_id = request
        .request_id
        .as_ref()
        .map(|id| id.request_id.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::Protocol("BidiAppend request_id is required".into()))?;
    // Cursor >= 3.30 sends the AgentClientMessage as raw protobuf bytes in
    // `data_binary` instead of the legacy hex string in `data`. The binary
    // payload may carry a transport envelope (Connect 5-byte frame, a varint
    // length prefix) or concatenate several messages, so probe the known
    // encodings before giving up.
    let raw: Vec<u8> = if !request.data_binary.is_empty() {
        request.data_binary.clone()
    } else if !request.data.is_empty() {
        hex::decode(&request.data)
            .map_err(|error| Error::Protocol(format!("invalid BidiAppend hex: {error}")))?
    } else {
        return Err(Error::Protocol(
            "BidiAppend contains no AgentClientMessage".into(),
        ));
    };
    tracing::warn!(
        "BidiAppend payload probe: data_len={} bin_len={} data_head={} bin_head={}",
        request.data.len(),
        raw.len(),
        &request.data[..request.data.len().min(120)],
        hex::encode(&raw[..raw.len().min(120)])
    );
    match decode_agent_message(&raw) {
        Ok(message) => {
            let message = normalize(message);
            Ok(DecodedAppend {
                request_id: request_id.into(),
                seqno: request.append_seqno,
                message,
                invocation_id: None,
                is_inference_control: false,
            })
        }
        Err(legacy_error) => {
            // Cursor >= 3.30 local loop multiplexes `RunInferenceClientMessage`
            // frames through BidiAppend.data_binary. These carry the
            // client-generated invocation id we must echo back on the SSE
            // bridge. Recognise them instead of rejecting with 400.
            if let Some(invocation_id) = decode_inference_invocation_id(&raw) {
                tracing::info!(
                    request_id,
                    invocation_id = %invocation_id,
                    "BidiAppend: decoded RunInferenceClientMessage.invoke_model"
                );
                return Ok(DecodedAppend {
                    request_id: request_id.into(),
                    seqno: request.append_seqno,
                    message: agent::AgentClientMessage::default(),
                    invocation_id: Some(invocation_id),
                    is_inference_control: true,
                });
            }
            tracing::warn!(
                "BidiAppend full payload hex ({} bytes): {}",
                raw.len(),
                hex::encode(&raw)
            );
            Err(legacy_error)
        }
    }
}

/// Try to decode a 3.30 `RunInferenceClientMessage` and pull the invocation id
/// out of `invoke_model` (field 2 -> field 1). Bare protobuf and the standard
/// 5-byte Connect frame envelope are both accepted.
fn decode_inference_invocation_id(raw: &[u8]) -> Option<String> {
    let candidates: [&[u8]; 2] = [
        raw,
        if raw.len() > 5 { &raw[5..] } else { &[] },
    ];
    for bytes in candidates {
        if bytes.is_empty() {
            continue;
        }
        if let Ok(message) = ai::RunInferenceClientMessage::decode(bytes) {
            if let Some(invoke) = message.invoke_model {
                if !invoke.invocation_id.is_empty() {
                    return Some(invoke.invocation_id);
                }
            }
        }
    }
    None
}

/// Cursor >= 3.30 moved the real conversation id to field 1 while field 5
/// now carries a harness marker ("agent"). Copy the UUID over so the rest
/// of the server keeps reading `conversation_id`.
fn normalize(mut message: agent::AgentClientMessage) -> agent::AgentClientMessage {
    if let Some(agent::agent_client_message::Message::RunRequest(request)) =
        message.message.as_mut()
    {
        if let Some(id_bytes) = request.conversation_id_330.take() {
            // 3.30 first turn sends an empty field; follow-up turns send a
            // nested non-UTF8 message. Only copy when it is a valid UTF8 UUID.
            if let Ok(id) = String::from_utf8(id_bytes) {
                if !id.is_empty() {
                    request.conversation_id = Some(id);
                }
            }
        }
    }
    message
}

fn try_decode(bytes: &[u8]) -> Option<agent::AgentClientMessage> {
    agent::AgentClientMessage::decode(bytes).ok()
}

fn decode_agent_message(raw: &[u8]) -> Result<agent::AgentClientMessage> {
    // 1. Bare protobuf.
    if let Some(message) = try_decode(raw) {
        return Ok(message);
    }
    // 2. Connect RPC envelope: flags byte + big-endian u32 length.
    if raw.len() > 5 {
        let len = u32::from_be_bytes([raw[1], raw[2], raw[3], raw[4]]) as usize;
        if len + 5 == raw.len() {
            if let Some(message) = try_decode(&raw[5..]) {
                tracing::warn!("BidiAppend payload had a 5-byte Connect frame");
                return Ok(message);
            }
        }
    }
    // 3. Varint length prefix followed by the message body.
    let mut value: usize = 0;
    let mut shift = 0usize;
    let mut prefix_end = 0usize;
    for (index, byte) in raw.iter().enumerate().take(5) {
        value |= ((byte & 0x7f) as usize) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            prefix_end = index + 1;
            break;
        }
    }
    if prefix_end > 0 && value + prefix_end == raw.len() {
        if let Some(message) = try_decode(&raw[prefix_end..]) {
            tracing::warn!("BidiAppend payload had a {prefix_end}-byte varint length prefix");
            return Ok(message);
        }
    }
    // 4. Concatenated messages: decode the first and report the rest.
    let probe_limit = raw.len().min(32 * 1024);
    for end in 1..=probe_limit {
        if let Some(first) = try_decode(&raw[..end]) {
            tracing::warn!(
                "BidiAppend payload had concatenated messages: first used {end} of {} bytes",
                raw.len()
            );
            return Ok(first);
        }
    }
    Err(Error::Protocol(
        "BidiAppend payload does not match any known encoding".into(),
    ))
}

pub async fn append(
    registry: &TransportRegistry,
    request: DecodedAppend,
    parent: Option<TransportParent>,
) -> Result<ai::BidiAppendResponse> {
    let replace_closing = request.model_id().is_some();
    let handle = registry
        .get_or_create_for_append(&request.request_id, replace_closing)
        .await?;
    let _admission = handle.admit()?;
    // 3.30 invoke_model frames carry the client-generated invocation id. Store
    // it on the handle for the SSE bridge; do not replay control frames into
    // the legacy run engine (the run was already driven by run_request).
    if let Some(invocation_id) = request.invocation_id.as_deref() {
        handle.set_invocation_id(invocation_id);
        return Ok(ai::BidiAppendResponse {});
    }
    if request.is_inference_control {
        return Ok(ai::BidiAppendResponse {});
    }
    if let Some(conversation_id) = request.conversation_id() {
        handle.set_conversation_id(conversation_id)?;
    }
    if let Some(parent) = parent {
        handle.set_parent(parent)?;
    }
    if matches!(
        request.message.message.as_ref(),
        Some(agent::agent_client_message::Message::ClientHeartbeat(_))
    ) {
        handle.emit(&events::heartbeat())?;
    }
    handle
        .command(TransportCommand::Append {
            seqno: request.seqno,
            message: Box::new(request.message),
        })
        .await?;
    Ok(ai::BidiAppendResponse {})
}
