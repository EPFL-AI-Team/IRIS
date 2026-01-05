import type { StateCreator } from "zustand";
import type {
  ServerConfig,
  ClientVideoConfig,
  PreviewFrame,
  CameraInfo,
  ConnectionStatus,
} from "../../types";

export interface CameraSlice {
  // Preview frame from backend USB camera
  previewFrame: PreviewFrame | null;
  setPreviewFrame: (frame: string, timestamp: number) => void;

  // Connection states (legacy - will be deprecated)
  previewConnection: ConnectionStatus;
  setPreviewConnection: (status: ConnectionStatus) => void;

  resultsConnection: ConnectionStatus;
  setResultsConnection: (status: ConnectionStatus) => void;

  browserStreamConnection: ConnectionStatus;
  setBrowserStreamConnection: (status: ConnectionStatus) => void;

  streamingServerStatus: ConnectionStatus;
  setStreamingServerStatus: (status: ConnectionStatus) => void;

  // Streaming state
  isStreaming: boolean;
  setIsStreaming: (active: boolean) => void;

  isCameraActive: boolean;
  setIsCameraActive: (active: boolean) => void;

  fps: number;
  setFps: (fps: number) => void;

  // Server configuration
  serverConfig: ServerConfig;
  setServerConfig: (config: ServerConfig) => void;

  // Client (browser) capture configuration
  clientVideoConfig: ClientVideoConfig;
  setClientVideoConfig: (config: ClientVideoConfig) => void;

  // Server cameras
  serverCameras: Array<CameraInfo>;
  setServerCameras: (cameras: Array<CameraInfo>) => void;

  selectedServerCamera: number;
  setSelectedServerCamera: (index: number) => void;

  // Client camera permission
  clientCameraPermission: "prompt" | "granted" | "denied";
  setClientCameraPermission: (status: "prompt" | "granted" | "denied") => void;

  // Client camera control
  clientCameraRequested: boolean;
  requestClientCamera: () => void;
  clearClientCameraRequest: () => void;

  // Manual reconnect triggers
  resultsReconnectToken: number;
  requestResultsReconnect: () => void;

  previewReconnectToken: number;
  requestPreviewReconnect: () => void;
}

export const createCameraSlice: StateCreator<
  CameraSlice,
  [],
  [],
  CameraSlice
> = (set) => ({
  // Preview frame from backend USB camera
  previewFrame: null,
  setPreviewFrame: (frame, timestamp) =>
    set({ previewFrame: { data: frame, timestamp } }),

  // Connection states
  previewConnection: "disconnected",
  setPreviewConnection: (status) => set({ previewConnection: status }),

  resultsConnection: "disconnected",
  setResultsConnection: (status) => set({ resultsConnection: status }),

  browserStreamConnection: "disconnected",
  setBrowserStreamConnection: (status) =>
    set({ browserStreamConnection: status }),

  streamingServerStatus: "disconnected",
  setStreamingServerStatus: (status) => set({ streamingServerStatus: status }),

  // Streaming state
  isStreaming: false,
  setIsStreaming: (active) => set({ isStreaming: active }),

  isCameraActive: false,
  setIsCameraActive: (active) => set({ isCameraActive: active }),

  fps: 0,
  setFps: (fps) => set({ fps }),

  // Server configuration
  serverConfig: {
    host: "localhost",
    port: 8005,
    endpoint: "/ws/stream",
  },
  setServerConfig: (config) => set({ serverConfig: config }),

  clientVideoConfig: {
    width: 640,
    height: 480,
    capture_fps: 10,
    jpeg_quality: 80,
    camera_index: 0,
  },
  setClientVideoConfig: (config) => set({ clientVideoConfig: config }),

  // Server cameras
  serverCameras: [],
  setServerCameras: (cameras) => set({ serverCameras: cameras }),
  selectedServerCamera: 0,
  setSelectedServerCamera: (index) => set({ selectedServerCamera: index }),

  // Client camera permission
  clientCameraPermission: "prompt",
  setClientCameraPermission: (status) =>
    set({ clientCameraPermission: status }),

  // Client camera control
  clientCameraRequested: false,
  requestClientCamera: () => set({ clientCameraRequested: true }),
  clearClientCameraRequest: () => set({ clientCameraRequested: false }),

  // Manual reconnect triggers
  resultsReconnectToken: 0,
  requestResultsReconnect: () =>
    set((state) => ({ resultsReconnectToken: state.resultsReconnectToken + 1 })),
  previewReconnectToken: 0,
  requestPreviewReconnect: () =>
    set((state) => ({ previewReconnectToken: state.previewReconnectToken + 1 })),
});
