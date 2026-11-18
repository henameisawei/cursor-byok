//! Serves the Cursor >= 3.30 agent run WebSocket transport.
//!
//! Cursor 3.30 no longer reads run output from `AgentService/RunSSE`. It dials
//! `wss://api2.cursor.sh/agent/v1/run` with subprotocol `cursor-agent-v1` and
//! speaks a private protobuf frame protocol (`agent.v1.AgentWebSocketServerFrame`):
//!
//! 1. client `hello{protocol_version: V1}` -> server `hello{protocol_version: V1}`
//! 2. client `run_start{identity{run_id, request_id}}` -> server `run_accepted`
//! 3. server `server_message{run_id, sequence, message}` frames, where `message`
//!    carries Connect envelopes (`AgentServerMessage`) exactly like the SSE body
//! 4. server `run_end{identity}` terminates the run
//!
//! The run itself is still started through `BidiService/BidiAppend`, so this
//! endpoint only attaches to the matching transport and replays its output.

use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::HeaderMap,
    response::Response,
};
use bytes::{Bytes, BytesMut};
use futures_util::{SinkExt, StreamExt};
use prost::Message as _;
use tokio::sync::mpsc;

use crate::cursor::{
    protocol::connect::END_STREAM_FLAG,
    transport::{TransportHandle, TransportRegistry},
};

const RUN_ID_HEADER: &str = "x-original-request-id";
const REQUEST_ID_HEADER: &str = "x-request-id";
const PROTOCOL_VERSION_V1: i32 = 1;

pub async fn upgrade(
    State(registry): State<TransportRegistry>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    // Cursor validates `socket.protocol === offered_subprotocol` right after the
    // 101 (openGeneration). If we do not echo one of its offers it fails the
    // whole pool generation with `handshake_failed` and reconnects forever.
    // Offers may be `cursor-agent-v1`, `cursor-agent-preconnect-v1`,
    // `cursor-run-inference-v1` or `cursor-run-inference-preconnect-v1`.
    let offered: Vec<String> = headers
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
                .collect()
        })
        .unwrap_or_default();
    tracing::info!(?offered, "agent websocket subprotocol offers");
    ws.protocols(offered)
        .on_upgrade(move |socket| serve(socket, registry, headers))
}

async fn serve(socket: WebSocket, registry: TransportRegistry, headers: HeaderMap) {
    let header_run_id = header_value(&headers, RUN_ID_HEADER);
    let header_request_id = header_value(&headers, REQUEST_ID_HEADER);
    tracing::info!(
        ?header_run_id,
        ?header_request_id,
        subprotocol = header_value(&headers, "sec-websocket-protocol"),
        "agent run websocket opened"
    );

    let (mut sink, mut input) = socket.split();
    let (outbound, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
    tokio::spawn(async move {
        while let Some(message) = outbound_rx.recv().await {
            let closing = matches!(message, Message::Close(_));
            if sink.send(message).await.is_err() || closing {
                return;
            }
        }
    });

    // --- handshake: hello, then run_start ---
    let (identity, handle) = loop {
        let Some(incoming) = input.next().await else {
            return;
        };
        let bytes = match incoming {
            Ok(Message::Binary(bytes)) => bytes,
            Ok(Message::Close(_)) | Err(_) => return,
            Ok(_) => continue,
        };
        let Ok(frame) = wire::AgentWebSocketClientFrame::decode(&bytes[..]) else {
            tracing::warn!(head = %head_hex(&bytes), "agent run websocket sent an unreadable frame");
            return;
        };
        match frame.frame {
            Some(wire::client_frame::Frame::Hello(hello)) => {
                tracing::info!(protocol_version = hello.protocol_version, "agent websocket hello");
                let reply = wire::AgentWebSocketServerFrame {
                    frame: Some(wire::server_frame::Frame::Hello(wire::AgentWebSocketHello {
                        protocol_version: PROTOCOL_VERSION_V1,
                    })),
                };
                if outbound
                    .send(Message::Binary(reply.encode_to_vec().into()))
                    .is_err()
                {
                    return;
                }
            }
            Some(wire::client_frame::Frame::RunStart(start)) => {
                let identity = start.identity.unwrap_or_default();
                tracing::info!(
                    run_id = %identity.run_id,
                    request_id = %identity.request_id,
                    "agent websocket run_start"
                );
                let Some(handle) = attach(&registry, &identity, header_request_id.as_deref()).await
                else {
                    tracing::warn!(
                        run_id = %identity.run_id,
                        request_id = %identity.request_id,
                        "agent websocket run has no local transport; rejecting"
                    );
                    let reject = wire::AgentWebSocketServerFrame {
                        frame: Some(wire::server_frame::Frame::RunRejected(
                            wire::AgentWebSocketRunRejected {
                                identity: Some(identity),
                                ..Default::default()
                            },
                        )),
                    };
                    let _ = outbound.send(Message::Binary(reject.encode_to_vec().into()));
                    let _ = outbound.send(Message::Close(None));
                    return;
                };
                let accepted = wire::AgentWebSocketServerFrame {
                    frame: Some(wire::server_frame::Frame::RunAccepted(
                        wire::AgentWebSocketRunAccepted {
                            identity: Some(identity.clone()),
                            headers: Vec::new(),
                        },
                    )),
                };
                if outbound
                    .send(Message::Binary(accepted.encode_to_vec().into()))
                    .is_err()
                {
                    return;
                }
                tracing::info!(
                    request_id = handle.request_id(),
                    "agent websocket accepted run"
                );
                break (identity, handle);
            }
            Some(other) => {
                tracing::info!(
                    frame = ?std::mem::discriminant(&other),
                    head = %head_hex(&bytes),
                    "agent websocket frame before run_start"
                );
            }
            None => return,
        }
    };

    // --- output pump: transport envelopes -> server_message frames ---
    let run_id = identity.run_id.clone();
    let pump_outbound = outbound.clone();
    let pump_handle = handle.clone();
    tokio::spawn(async move {
        let mut output = pump_handle.subscribe();
        let mut sequence: i64 = 0;
        while let Some(chunk) = output.recv().await {
            let Some(envelopes) = split_envelopes(&chunk) else {
                continue;
            };
            for (flags, payload) in envelopes {
                if flags & END_STREAM_FLAG != 0 {
                    let end = wire::AgentWebSocketServerFrame {
                        frame: Some(wire::server_frame::Frame::RunEnd(wire::AgentWebSocketRunEnd {
                            identity: Some(wire::AgentWebSocketRunIdentity {
                                run_id: run_id.clone(),
                                request_id: String::new(),
                            }),
                            trailers: Vec::new(),
                            error: None,
                        })),
                    };
                    let _ = pump_outbound.send(Message::Binary(end.encode_to_vec().into()));
                    let _ = pump_outbound.send(Message::Close(None));
                    return;
                }
                let mut message = BytesMut::with_capacity(5 + payload.len());
                message.extend_from_slice(&(payload.len() as u32).to_be_bytes()[..]);
                let mut envelope = BytesMut::with_capacity(5 + payload.len());
                envelope.extend_from_slice(&[0u8]);
                envelope.extend_from_slice(&message);
                envelope.extend_from_slice(&payload);
                let frame = wire::AgentWebSocketServerFrame {
                    frame: Some(wire::server_frame::Frame::ServerMessage(
                        wire::AgentWebSocketServerMessage {
                            run_id: run_id.clone(),
                            sequence,
                            message: envelope.to_vec(),
                        },
                    )),
                };
                sequence += 1;
                if pump_outbound
                    .send(Message::Binary(frame.encode_to_vec().into()))
                    .is_err()
                {
                    return;
                }
            }
        }
    });

    // --- run phase: only inbound control frames remain ---
    while let Some(incoming) = input.next().await {
        match incoming {
            Ok(Message::Binary(bytes)) => match wire::AgentWebSocketClientFrame::decode(&bytes[..]) {
                Ok(frame) => match frame.frame {
                    Some(wire::client_frame::Frame::ClientMessage(message)) => {
                        tracing::info!(
                            run_id = %message.run_id,
                            sequence = message.sequence,
                            len = message.message.len(),
                            head = %head_hex(&message.message),
                            "agent websocket client message"
                        );
                    }
                    Some(wire::client_frame::Frame::CancelRun(cancel)) => {
                        tracing::info!(run_id = %cancel.run_id, "agent websocket cancel run");
                        handle.disconnect().await;
                        return;
                    }
                    Some(wire::client_frame::Frame::HalfClose(close)) => {
                        tracing::info!(run_id = %close.run_id, "agent websocket half close");
                    }
                    _ => {}
                },
                Err(error) => {
                    tracing::warn!(%error, head = %head_hex(&bytes), "agent websocket unreadable client frame");
                }
            },
            Ok(Message::Close(_)) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

async fn attach(
    registry: &TransportRegistry,
    identity: &wire::AgentWebSocketRunIdentity,
    header_request_id: Option<&str>,
) -> Option<TransportHandle> {
    let mut candidates: Vec<&str> = Vec::new();
    for candidate in [
        Some(identity.request_id.as_str()),
        Some(identity.run_id.as_str()),
        header_request_id,
    ]
    .into_iter()
    .flatten()
    {
        if !candidate.is_empty() && !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    for _ in 0..20 {
        for candidate in &candidates {
            if let Some(handle) = registry.local(candidate).await {
                return Some(handle);
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let fallback = registry.latest_local().await;
    if fallback.is_some() {
        tracing::warn!(
            candidates = ?candidates,
            "agent websocket identity is unknown; attaching to newest local run"
        );
    }
    fallback
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn head_hex(bytes: &[u8]) -> String {
    let head = &bytes[..bytes.len().min(48)];
    let mut text = head
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if bytes.len() > head.len() {
        text.push('…');
    }
    text
}

fn split_envelopes(chunk: &Bytes) -> Option<Vec<(u8, Bytes)>> {
    let mut frames = Vec::new();
    let mut body: &[u8] = chunk;
    while !body.is_empty() {
        if body.len() < 5 {
            return None;
        }
        let flags = body[0];
        let length = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        body = &body[5..];
        if body.len() < length {
            return None;
        }
        frames.push((flags, Bytes::copy_from_slice(&body[..length])));
        body = &body[length..];
    }
    Some(frames)
}

/// Wire types of the private `agent.v1` WebSocket frame protocol. Only the
/// fields Cursor actually validates are modelled.
mod wire {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketHello {
        #[prost(int32, tag = "1")]
        pub protocol_version: i32,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketRunIdentity {
        #[prost(string, tag = "1")]
        pub run_id: String,
        #[prost(string, tag = "2")]
        pub request_id: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketHeader {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(string, tag = "2")]
        pub value: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketRunStart {
        #[prost(message, optional, tag = "1")]
        pub identity: Option<AgentWebSocketRunIdentity>,
        #[prost(message, repeated, tag = "2")]
        pub headers: Vec<AgentWebSocketHeader>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketClientMessage {
        #[prost(string, tag = "1")]
        pub run_id: String,
        #[prost(int64, tag = "2")]
        pub sequence: i64,
        #[prost(bytes = "vec", tag = "3")]
        pub message: Vec<u8>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketClientHalfClose {
        #[prost(string, tag = "1")]
        pub run_id: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketCancelRun {
        #[prost(string, tag = "1")]
        pub run_id: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketClientFrame {
        #[prost(oneof = "client_frame::Frame", tags = "1, 2, 3, 4, 5")]
        pub frame: Option<client_frame::Frame>,
    }

    pub mod client_frame {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Frame {
            #[prost(message, tag = "1")]
            Hello(super::AgentWebSocketHello),
            #[prost(message, tag = "2")]
            RunStart(super::AgentWebSocketRunStart),
            #[prost(message, tag = "3")]
            ClientMessage(super::AgentWebSocketClientMessage),
            #[prost(message, tag = "4")]
            HalfClose(super::AgentWebSocketClientHalfClose),
            #[prost(message, tag = "5")]
            CancelRun(super::AgentWebSocketCancelRun),
        }
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketRunAccepted {
        #[prost(message, optional, tag = "1")]
        pub identity: Option<AgentWebSocketRunIdentity>,
        #[prost(message, repeated, tag = "2")]
        pub headers: Vec<AgentWebSocketHeader>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketServerMessage {
        #[prost(string, tag = "1")]
        pub run_id: String,
        #[prost(int64, tag = "2")]
        pub sequence: i64,
        #[prost(bytes = "vec", tag = "3")]
        pub message: Vec<u8>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketConnectErrorDetail {
        #[prost(string, tag = "1")]
        pub type_name: String,
        #[prost(bytes = "vec", tag = "2")]
        pub binary_value: Vec<u8>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketConnectError {
        #[prost(int32, tag = "1")]
        pub code: i32,
        #[prost(string, tag = "2")]
        pub raw_message: String,
        #[prost(message, repeated, tag = "3")]
        pub metadata: Vec<AgentWebSocketHeader>,
        #[prost(message, repeated, tag = "4")]
        pub details: Vec<AgentWebSocketConnectErrorDetail>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketRunEnd {
        #[prost(message, optional, tag = "1")]
        pub identity: Option<AgentWebSocketRunIdentity>,
        #[prost(message, repeated, tag = "2")]
        pub trailers: Vec<AgentWebSocketHeader>,
        #[prost(message, optional, tag = "3")]
        pub error: Option<AgentWebSocketConnectError>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketRunRejected {
        #[prost(message, optional, tag = "1")]
        pub identity: Option<AgentWebSocketRunIdentity>,
        #[prost(message, optional, tag = "2")]
        pub error: Option<AgentWebSocketConnectError>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AgentWebSocketServerFrame {
        #[prost(oneof = "server_frame::Frame", tags = "1, 2, 3, 4, 5")]
        pub frame: Option<server_frame::Frame>,
    }

    pub mod server_frame {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Frame {
            #[prost(message, tag = "1")]
            Hello(super::AgentWebSocketHello),
            #[prost(message, tag = "2")]
            RunAccepted(super::AgentWebSocketRunAccepted),
            #[prost(message, tag = "3")]
            ServerMessage(super::AgentWebSocketServerMessage),
            #[prost(message, tag = "4")]
            RunEnd(super::AgentWebSocketRunEnd),
            #[prost(message, tag = "5")]
            RunRejected(super::AgentWebSocketRunRejected),
        }
    }
}
