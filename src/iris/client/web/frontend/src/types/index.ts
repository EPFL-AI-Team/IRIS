// Connection status for WebSocket connections
export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

// Camera mode
export type CameraMode = "client" | "server";

// Server configuration
export interface ServerConfig {
  host: string;
  port: number;
  endpoint: string;
}

export interface ClientVideoConfig {
  width: number;
  height: number;
  capture_fps: number;
  jpeg_quality: number;
  camera_index: number;
}

// Log entry
export type LogLevel = "INFO" | "WARNING" | "ERROR" | "DEBUG";

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  message: string;
}

// Result item (can be pending or completed)
export interface ResultItem {
  id: string;
  job_id: string;
  timestamp: Date;
  status: "pending" | "processing" | "completed";
  result?: string;
  frames_processed?: number;
  inference_time?: number;
  pendingDetails?: string;
}

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
    server: ServerConfig;
    video: ClientVideoConfig;
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

// Frame data for browser camera streaming
export interface FrameData {
  frame: string; // base64 encoded JPEG
  frame_id: number;
  timestamp: number;
  fps: number;
  measured_fps?: number;
}

// Camera info from server
export interface CameraInfo {
  index: number;
  name: string;
  resolution: string;
}

// API response types
export interface ApiResponse<T = unknown> {
  status: "ok" | "error";
  message?: string;
  data?: T;
}

// Toast notification
export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

// Tab type for main navigation
export type ActiveTab = "live" | "analysis";

// Preview frame from WebSocket
export interface PreviewFrame {
  data: string; // base64 JPEG
  timestamp: number;
}

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

// Analysis types
export * from "./analysis";
