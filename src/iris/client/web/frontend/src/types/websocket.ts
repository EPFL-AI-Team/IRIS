// WebSocket connection and message types

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

// WebSocket message types from backend
export interface KeepaliveMessage {
  type: "keepalive";
  timestamp: number;
}

export interface BatchSubmittedMessage {
  type: "batch_submitted";
  job_id: string;
  timestamp: number;
  status: string;
}

export interface LogMessage {
  type: "log";
  job_id: string;
  message: string;
  timestamp: number;
}

export interface ResultMessage {
  type: "result";
  job_id: string;
  status: string;
  result: string;
  frames_processed: number;
  inference_time: number;
  buffer_size?: number;
  overlap_frames?: number;
  client_fps?: number;
  timestamp: number;
}

export interface StatusUpdateMessage {
  type: "status_update";
  camera_active: boolean;
  streaming_active: boolean;
  streaming_server_status: ConnectionStatus;
  fps: number;
  config: {
    server: {
      host: string;
      port: number;
      endpoint: string;
    };
    video: {
      width: number;
      height: number;
      capture_fps: number;
      jpeg_quality: number;
      camera_index: number;
    };
  };
  timestamp: number;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

// Union of all WebSocket message types
export type WebSocketMessage =
  | KeepaliveMessage
  | BatchSubmittedMessage
  | LogMessage
  | ResultMessage
  | StatusUpdateMessage
  | ErrorMessage;

// Simplified WebSocket message types (new architecture)
export interface SessionInfoMessage {
  type: "session_info";
  session_id: string;
  config: Record<string, unknown>;
}

export interface PreviewFrameMessage {
  type: "preview_frame";
  frame: string;
  timestamp: number;
}

export interface MetricsMessage {
  type: "metrics";
  elapsed_seconds: number;
  segments_processed: number;
  segments_total?: number;
  queue_depth: number;
  processing_rate: number;
}

export interface ServerStatusMessage {
  type: "server_status";
  alive: boolean;
  queue_depth?: number;
}

// Union of new WebSocket message types
export type ClientWebSocketMessage =
  | SessionInfoMessage
  | PreviewFrameMessage
  | ResultMessage
  | MetricsMessage
  | ServerStatusMessage
  | ErrorMessage;
