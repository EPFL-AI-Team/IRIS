// Re-export all types for backward compatibility

// Camera types
export type {
  ServerConfig,
  ClientVideoConfig,
  CameraInfo,
  PreviewFrame,
  FrameData,
} from "./camera";

// WebSocket types
export type {
  ConnectionStatus,
  KeepaliveMessage,
  BatchSubmittedMessage,
  LogMessage,
  ResultMessage,
  StatusUpdateMessage,
  ErrorMessage,
  WebSocketMessage,
  SessionInfoMessage,
  PreviewFrameMessage,
  MetricsMessage,
  ServerStatusMessage,
  ClientWebSocketMessage,
} from "./websocket";

// UI types
export type {
  LogLevel,
  LogEntry,
  ResultItem,
  ToastType,
  Toast,
  ActiveTab,
  ApiResponse,
} from "./ui";

// Analysis types (already in separate file)
export * from "./analysis";

