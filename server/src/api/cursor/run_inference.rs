//! Bridges the Cursor 3.30 managed-local-run output transport.
//!
//! Cursor >= 3.30 no longer reads run output from `AgentService/RunSSE`. The
//! agent host opens `aiserver.v1.InferenceService/RunInferenceSSE` with a
//! `BidiRequestId` and expects a stream of `RunInferenceServerMessage` frames:
//!
//!   run_ready  -> invocation_response{text_part / tool_call_part}* -> invocation_end
//!
//! byok's run engine still produces legacy `agent.v1.AgentServerMessage`
//! Connect envelopes into the transport. This module decodes those envelopes
//! and re-frames them as `RunInferenceServerMessage` so the new client can
//! render BYOK model output.

use axum::{
    body::Body,
    http::{header, HeaderValue, Response},
};
use bytes::Bytes;
use prost::Message;
use tokio::sync::mpsc;

use crate::{
    cursor::{
        protocol::{
            connect::{self, END_STREAM_FLAG},
            proto::agent::v1 as agent,
        },
        transport::{TransportHandle, TransportRegistry},
    },
    Result,
};

/// Model id advertised in `run_ready.resolved_model`. The client already knows
/// what it requested; it only needs a non-empty id here to build its config.
const FALLBACK_MODEL_ID: &str = "byok";

pub async fn stream(
    registry: &TransportRegistry,
    request_id: &str,
    model_id: Option<String>,
) -> Result<Response<Body>> {
    let handle = registry.get_or_create(request_id).await?;
    let receiver = handle.subscribe();
    let body_stream = inference_body_stream(receiver, handle, model_id);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = axum::http::StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response.headers_mut().insert(
        "connect-protocol-version",
        HeaderValue::from_static("1"),
    );
    Ok(response)
}

/// Wait for the client-generated invocation id delivered on BidiAppend, with a
/// bounded fallback derived from the request id. The client builds its id as
/// `${requestId}-${count}-${uuid}`; a fallback with the right prefix is the
/// best we can do if the invoke_model frame never arrives.
async fn resolve_invocation_id(handle: &TransportHandle) -> String {
    for _ in 0..40 {
        if let Some(id) = handle.invocation_id() {
            return id.to_string();
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    format!("{}-0-byok", handle.request_id())
}

fn encode_frame<M: Message>(message: &M) -> Bytes {
    connect::encode_message(message).unwrap_or_else(|_| Bytes::new())
}

fn server_frame(frame: wire::server_message::Frame) -> Bytes {
    encode_frame(&wire::RunInferenceServerMessage {
        frame: Some(frame),
    })
}

fn inference_body_stream(
    mut receiver: mpsc::UnboundedReceiver<Bytes>,
    handle: TransportHandle,
    model_id: Option<String>,
) -> impl tokio_stream::Stream<Item = std::result::Result<Bytes, std::convert::Infallible>> {
    async_stream::stream! {
        tracing::info!(
            request_id = %handle.request_id(),
            "run inference bridge: stream polled (client is reading body)"
        );
        // 1. run_ready. Cursor >= 3.30's managed local loop refuses to run unless
        //    run_ready carries field 4 `prompt_model_metadata`; without it the
        //    client throws `metadata-unavailable` ("server sent no prompt
        //    metadata") before it ever renders our output. Mirror the legacy
        //    behaviour by advertising a generic OpenAI-compatible model family
        //    so the client can construct its prompt.
        let ready = wire::RunInferenceRunReady {
            resolved_model: Some(wire::InferenceRequestedModel {
                model_id: model_id.clone().unwrap_or_else(|| FALLBACK_MODEL_ID.to_string()),
                built_in_model: true,
                ..Default::default()
            }),
            prompt_model_metadata: Some(wire::RunInferencePromptModelMetadata {
                // `vendor` must be one of anthropic/openai/google/xai/zai/cursor/unknown
                // and `prompt_version` one of dsv3-1018/dsv3-1205/cursor-0226/latest/
                // gpt5-codex/codex-cloud/haiku, otherwise the client rejects the
                // model family. DeepSeek is OpenAI-compatible, so use the plain
                // openai vendor with the generic "latest" prompt path.
                vendor: "openai".to_string(),
                prompt_version: "latest".to_string(),
                self_identity_name: String::new(),
                ..Default::default()
            }),
            // Tell the client it may send heartbeats too. More importantly we
            // emit our own keepalives below, since the client's local-loop stall
            // detector is driven by inbound frames and aborts after 90s of silence.
            accepts_client_heartbeat: true,
            ..Default::default()
        };
        yield Ok(server_frame(wire::server_message::Frame::RunReady(ready)));

        // The client generates its own invocation id (`${requestId}-${count}-${uuid}`)
        // and delivers it via a RunInferenceClientMessage.invoke_model frame on
        // BidiAppend shortly after this stream opens. We must echo that exact id
        // back or the client's `invocations.get(id)` lookup drops every frame.
        // Wait briefly for it to arrive (observed ~30ms after stream open).
        let invocation_id = resolve_invocation_id(&handle).await;
        tracing::info!(
            request_id = %handle.request_id(),
            invocation_id = %invocation_id,
            "run inference bridge: using invocation id"
        );
        let mut saw_text = false;
        let mut finished = false;
        let mut chunks_seen: u64 = 0;
        let mut frames_decoded: u64 = 0;
        let mut messages_decoded: u64 = 0;
        let mut responses_emitted: u64 = 0;

        // The client's local-loop stall detector aborts the whole attempt after
        // 90s with no inbound frame (30s advisory). Our run engine can sit silent
        // for 10-21s waiting on the state-ready barrier before emitting its first
        // token, during which the old code sent nothing and the client aborted
        // even though the model eventually finished. Keep the stream alive with
        // periodic heartbeat frames (RunInferenceServerMessage tag 1) while we
        // wait for real output.
        let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(5));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            let mut ping = false;
            let chunk = tokio::select! {
                chunk = receiver.recv() => chunk,
                _ = heartbeat.tick() => { ping = true; None },
            };
            if ping {
                let hb = wire::RunInferenceHeartbeat {
                    accepts_speculative_invocations: false,
                    accepts_compressed_speculative_invocations: false,
                };
                yield Ok(server_frame(wire::server_message::Frame::Heartbeat(hb)));
                continue;
            }
            let Some(chunk) = chunk else { break; };
            chunks_seen += 1;
            if chunk.first().is_some_and(|flags| flags & END_STREAM_FLAG != 0) {
                tracing::info!(
                    request_id = %handle.request_id(),
                    chunks_seen, frames_decoded, messages_decoded, responses_emitted,
                    "run inference bridge: end-stream, stopping"
                );
                break;
            }
            let Ok(frames) = connect::decode_frames(&chunk) else {
                tracing::warn!(
                    request_id = %handle.request_id(),
                    chunk_len = chunk.len(),
                    "run inference bridge: chunk failed connect::decode_frames"
                );
                continue;
            };
            for (flags, payload) in frames {
                if flags & END_STREAM_FLAG != 0 {
                    continue;
                }
                frames_decoded += 1;
                let Ok(message) = agent::AgentServerMessage::decode(&payload[..]) else {
                    tracing::warn!(
                        request_id = %handle.request_id(),
                        payload_len = payload.len(),
                        "run inference bridge: frame failed AgentServerMessage::decode"
                    );
                    continue;
                };
                messages_decoded += 1;
                // PROBE: record exactly which message arrived on the bridge's hub.
                tracing::info!(
                    request_id = %handle.request_id(),
                    kind = %describe_agent_message(&message),
                    "run inference bridge: PROBE decoded frame"
                );
                let Some(agent::agent_server_message::Message::InteractionUpdate(update)) =
                    message.message
                else {
                    continue;
                };
                let Some(part) = byok_update_to_response(&update, &mut saw_text)
                else {
                    continue;
                };
                responses_emitted += 1;
                let response = wire::RunInferenceInvocationResponse {
                    invocation_id: invocation_id.clone(),
                    response: Some(part),
                };
                yield Ok(server_frame(wire::server_message::Frame::InvocationResponse(response)));
            }
        }
        tracing::info!(
            request_id = %handle.request_id(),
            chunks_seen, frames_decoded, messages_decoded, responses_emitted,
            "run inference bridge: receiver drained"
        );

        // 2. flush a final text_part so the client closes the current text block
        if saw_text {
            let final_text = wire::InferenceStreamResponse {
                response: Some(wire::inference_stream_response::Response::TextPart(
                    wire::InferenceTextStreamPart {
                        text: String::new(),
                        is_final: true,
                    },
                )),
            };
            let response = wire::RunInferenceInvocationResponse {
                invocation_id: invocation_id.clone(),
                response: Some(final_text),
            };
            yield Ok(server_frame(wire::server_message::Frame::InvocationResponse(response)));
        }

        // 3. invocation_end then stream end
        let end = wire::RunInferenceInvocationEnd {
            invocation_id: invocation_id.clone(),
            error: None,
        };
        yield Ok(server_frame(wire::server_message::Frame::InvocationEnd(end)));
        finished = true;

        if !finished {
            handle.disconnect().await;
        }
    }
}

/// PROBE: human-readable tag of an AgentServerMessage, used only for diagnosis.
fn describe_agent_message(message: &agent::AgentServerMessage) -> String {
    use agent::agent_server_message::Message as M;
    match message.message.as_ref() {
        None => "none".to_string(),
        Some(M::InteractionUpdate(update)) => {
            use agent::interaction_update::Message as U;
            let sub = match update.message.as_ref() {
                Some(U::TextDelta(_)) => "text_delta",
                Some(U::ThinkingDelta(_)) => "thinking_delta",
                Some(U::PartialToolCall(_)) => "partial_tool_call",
                Some(U::ToolCallDelta(_)) => "tool_call_delta",
                Some(U::ToolCallStarted(_)) => "tool_call_started",
                Some(U::ToolCallCompleted(_)) => "tool_call_completed",
                Some(U::ThinkingCompleted(_)) => "thinking_completed",
                Some(U::UserMessageAppended(_)) => "user_message_appended",
                Some(U::TokenDelta(_)) => "token_delta",
                Some(U::Summary(_)) => "summary",
                Some(U::SummaryStarted(_)) => "summary_started",
                Some(U::SummaryCompleted(_)) => "summary_completed",
                Some(U::ShellOutputDelta(_)) => "shell_output_delta",
                Some(U::Heartbeat(_)) => "heartbeat",
                Some(U::TurnEnded(_)) => "turn_ended",
                Some(U::StepStarted(_)) => "step_started",
                Some(U::StepCompleted(_)) => "step_completed",
                Some(U::PromptSuggestion(_)) => "prompt_suggestion",
                Some(U::PostRequestPrompt(_)) => "post_request_prompt",
                Some(U::ActiveBranchChange(_)) => "active_branch_change",
                Some(U::FeedbackRequest(_)) => "feedback_request",
                Some(U::ResponseComparison(_)) => "response_comparison",
                Some(U::ContextInjectionState(_)) => "context_injection_state",
                None => "interaction_none",
            };
            format!("interaction_update/{sub}")
        }
        Some(M::ExecServerMessage(_)) => "exec_server_message".to_string(),
        Some(M::ExecServerControlMessage(_)) => "exec_server_control_message".to_string(),
        Some(M::ConversationCheckpointUpdate(_)) => "conversation_checkpoint_update".to_string(),
        Some(M::KvServerMessage(_)) => "kv_server_message".to_string(),
        Some(M::InteractionQuery(_)) => "interaction_query".to_string(),
    }
}

fn byok_update_to_response(
    update: &agent::InteractionUpdate,
    saw_text: &mut bool,
) -> Option<wire::InferenceStreamResponse> {
    use agent::interaction_update::Message as U;
    let response = match update.message.as_ref()? {
        U::TextDelta(delta) if !delta.text.is_empty() => {
            *saw_text = true;
            wire::inference_stream_response::Response::TextPart(wire::InferenceTextStreamPart {
                text: delta.text.clone(),
                is_final: false,
            })
        }
        U::ThinkingDelta(delta) if !delta.text.is_empty() => {
            wire::inference_stream_response::Response::ThinkingPart(wire::InferenceThinkingStreamPart {
                text: delta.text.clone(),
                is_final: false,
                ..Default::default()
            })
        }
        U::ToolCallStarted(started) => {
            wire::inference_stream_response::Response::ToolCallPart(wire::InferenceToolCallStreamPart {
                tool_call_id: started.call_id.clone(),
                tool_name: String::new(),
                args: String::new(),
                is_complete: false,
                tool_index: None,
                call_signature: None,
            })
        }
        U::PartialToolCall(partial) => {
            wire::inference_stream_response::Response::ToolCallPart(wire::InferenceToolCallStreamPart {
                tool_call_id: partial.call_id.clone(),
                tool_name: String::new(),
                args: partial.args_text_delta.clone(),
                is_complete: false,
                tool_index: None,
                call_signature: None,
            })
        }
        U::ToolCallCompleted(completed) => {
            wire::inference_stream_response::Response::ToolCallPart(wire::InferenceToolCallStreamPart {
                tool_call_id: completed.call_id.clone(),
                tool_name: String::new(),
                args: String::new(),
                is_complete: true,
                tool_index: None,
                call_signature: None,
            })
        }
        U::TokenDelta(_) => return None,
        _ => return None,
    };
    Some(wire::InferenceStreamResponse {
        response: Some(response),
    })
}

/// Wire types of the `aiserver.v1.InferenceService` protocol that Cursor 3.30
/// uses for BYOK run output. Only fields we produce are modelled.
mod wire {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceRequestedModel {
        #[prost(string, tag = "1")]
        pub model_id: String,
        #[prost(bool, tag = "2")]
        pub max_mode: bool,
        #[prost(message, repeated, tag = "3")]
        pub parameters: Vec<ModelParameter>,
        #[prost(bool, tag = "4")]
        pub built_in_model: bool,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ModelParameter {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(string, tag = "2")]
        pub value: String,
    }

    /// Mirrors aiserver.v1.RunInferencePromptModelMetadata. Only the fields
    /// required by the client's `promptModelInfoFromWire` / `resolvedModelMetadataFromWire`
    /// are modelled; the rest stay at their proto3 defaults.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferencePromptModelMetadata {
        #[prost(string, tag = "1")]
        pub vendor: String,
        #[prost(string, tag = "2")]
        pub prompt_version: String,
        #[prost(string, optional, tag = "30")]
        pub reasoning_effort: Option<String>,
        #[prost(bool, tag = "31")]
        pub use_dsv3_harness: bool,
        #[prost(string, tag = "35")]
        pub persona: String,
        #[prost(string, tag = "44")]
        pub self_identity_name: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferenceRunReady {
        #[prost(message, optional, tag = "1")]
        pub resolved_model: Option<InferenceRequestedModel>,
        #[prost(bool, tag = "2")]
        pub supports_self_summary: bool,
        #[prost(string, optional, tag = "3")]
        pub routed_model_display_name: Option<String>,
        #[prost(message, optional, tag = "4")]
        pub prompt_model_metadata: Option<RunInferencePromptModelMetadata>,
        #[prost(bool, tag = "7")]
        pub accepts_client_heartbeat: bool,
        #[prost(bool, tag = "8")]
        pub accepts_agent_events: bool,
        #[prost(bool, tag = "15")]
        pub accepts_tool_call_events: bool,
        #[prost(bytes = "vec", tag = "17")]
        pub config_fingerprint: Vec<u8>,
        #[prost(bool, tag = "18")]
        pub accepts_delta_invocations: bool,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceTextStreamPart {
        #[prost(string, tag = "1")]
        pub text: String,
        #[prost(bool, tag = "2")]
        pub is_final: bool,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceToolCallStreamPart {
        #[prost(string, tag = "1")]
        pub tool_call_id: String,
        #[prost(string, tag = "2")]
        pub tool_name: String,
        #[prost(string, tag = "3")]
        pub args: String,
        #[prost(bool, tag = "4")]
        pub is_complete: bool,
        #[prost(int32, optional, tag = "5")]
        pub tool_index: Option<i32>,
        #[prost(string, optional, tag = "6")]
        pub call_signature: Option<String>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceThinkingStreamPart {
        #[prost(string, tag = "1")]
        pub text: String,
        #[prost(string, optional, tag = "2")]
        pub signature: Option<String>,
        #[prost(bool, tag = "3")]
        pub is_final: bool,
        #[prost(bool, optional, tag = "4")]
        pub is_already_summarized: Option<bool>,
        #[prost(string, optional, tag = "5")]
        pub model_name: Option<String>,
        #[prost(bytes = "vec", optional, tag = "6")]
        pub redacted_data: Option<Vec<u8>>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceUsageInfo {
        #[prost(int32, tag = "1")]
        pub prompt_tokens: i32,
        #[prost(int32, tag = "2")]
        pub completion_tokens: i32,
        #[prost(int32, optional, tag = "3")]
        pub total_tokens: Option<i32>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceStreamResponse {
        #[prost(oneof = "inference_stream_response::Response", tags = "1, 2, 3, 9")]
        pub response: Option<inference_stream_response::Response>,
    }

    pub mod inference_stream_response {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Response {
            #[prost(message, tag = "1")]
            TextPart(super::InferenceTextStreamPart),
            #[prost(message, tag = "2")]
            ToolCallPart(super::InferenceToolCallStreamPart),
            #[prost(message, tag = "3")]
            Usage(super::InferenceUsageInfo),
            #[prost(message, tag = "9")]
            ThinkingPart(super::InferenceThinkingStreamPart),
        }
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferenceInvocationResponse {
        #[prost(string, tag = "1")]
        pub invocation_id: String,
        #[prost(message, optional, tag = "2")]
        pub response: Option<InferenceStreamResponse>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferenceInvocationEnd {
        #[prost(string, tag = "1")]
        pub invocation_id: String,
        #[prost(message, optional, tag = "2")]
        pub error: Option<InferenceStreamError>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceStreamError {
        #[prost(string, tag = "1")]
        pub message: String,
        #[prost(string, tag = "2")]
        pub code: String,
    }

    /// Heartbeat frame (aiserver.v1.RunInferenceHeartbeat, tag 1). The client's
    /// local-loop stall detector counts inbound frames; if none arrive within
    /// its threshold (30s advisory / 90s fatal) it aborts the whole attempt even
    /// though the model eventually finished. Emitting a keepalive heartbeat while
    /// we wait for the run engine keeps that detector satisfied.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferenceHeartbeat {
        #[prost(bool, tag = "1")]
        pub accepts_speculative_invocations: bool,
        #[prost(bool, tag = "2")]
        pub accepts_compressed_speculative_invocations: bool,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct RunInferenceServerMessage {
        #[prost(oneof = "server_message::Frame", tags = "1, 2, 3, 4")]
        pub frame: Option<server_message::Frame>,
    }

    pub mod server_message {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Frame {
            #[prost(message, tag = "1")]
            Heartbeat(super::RunInferenceHeartbeat),
            #[prost(message, tag = "2")]
            RunReady(super::RunInferenceRunReady),
            #[prost(message, tag = "3")]
            InvocationResponse(super::RunInferenceInvocationResponse),
            #[prost(message, tag = "4")]
            InvocationEnd(super::RunInferenceInvocationEnd),
        }
    }
}
