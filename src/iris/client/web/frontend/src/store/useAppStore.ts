import { create } from "zustand";
import type {
  CameraMode,
  ConnectionStatus,
  ClientVideoConfig,
  LogEntry,
  LogLevel,
  ResultItem,
  ServerConfig,
  SSHTunnelConfig,
} from "../types";

const MAX_LOG_ENTRIES = 100;
const MAX_RESULTS = 50;

interface AppState {
  // Camera mode
  cameraMode: CameraMode;
  setCameraMode: (mode: CameraMode) => void;

  // Connection states
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

  sshTunnelConfig: SSHTunnelConfig;
  setSSHTunnelConfig: (config: SSHTunnelConfig) => void;

  // Client (browser) capture configuration (from backend config.yaml)
  clientVideoConfig: ClientVideoConfig;
  setClientVideoConfig: (config: ClientVideoConfig) => void;

  // Results
  results: ResultItem[];
  addResult: (result: ResultItem) => void;
  updateResult: (jobId: string, updates: Partial<ResultItem>) => void;
  clearResults: () => void;

  // Logs
  logs: LogEntry[];
  addLog: (message: string, level?: LogLevel) => void;
  clearLogs: () => void;

  // Server cameras (for server camera mode)
  serverCameras: Array<{ index: number; name: string; resolution: string }>;
  setServerCameras: (
    cameras: Array<{ index: number; name: string; resolution: string }>
  ) => void;
  selectedServerCamera: number;
  setSelectedServerCamera: (index: number) => void;

  // Client camera permission
  clientCameraPermission: "prompt" | "granted" | "denied";
  setClientCameraPermission: (status: "prompt" | "granted" | "denied") => void;

  // Client camera control (trigger start from other components)
  clientCameraRequested: boolean;
  requestClientCamera: () => void;
  clearClientCameraRequest: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Camera mode - default to client (browser camera)
  cameraMode: "client",
  setCameraMode: (mode) => set({ cameraMode: mode }),

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

  sshTunnelConfig: {
    remote_host: "",
  },
  setSSHTunnelConfig: (config) => set({ sshTunnelConfig: config }),

  clientVideoConfig: {
    width: 640,
    height: 480,
    capture_fps: 10,
    jpeg_quality: 80,
    camera_index: 0,
  },
  setClientVideoConfig: (config) => set({ clientVideoConfig: config }),

  // Results
  results: [],
  addResult: (result) =>
    set((state) => {
      // Check if result already exists (by job_id)
      const existingIndex = state.results.findIndex(
        (r) => r.job_id === result.job_id
      );
      if (existingIndex >= 0) {
        // Update existing result
        const newResults = [...state.results];
        newResults[existingIndex] = { ...newResults[existingIndex], ...result };
        return { results: newResults };
      }
      // Add new result, limit total
      const newResults = [...state.results, result];
      if (newResults.length > MAX_RESULTS) {
        newResults.shift();
      }
      return { results: newResults };
    }),
  updateResult: (jobId, updates) =>
    set((state) => {
      const newResults = state.results.map((r) =>
        r.job_id === jobId ? { ...r, ...updates } : r
      );
      return { results: newResults };
    }),
  clearResults: () => set({ results: [] }),

  // Logs
  logs: [],
  addLog: (message, level = "INFO") =>
    set((state) => {
      const newLog: LogEntry = {
        id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        timestamp: new Date(),
        level,
        message,
      };
      const newLogs = [...state.logs, newLog];
      if (newLogs.length > MAX_LOG_ENTRIES) {
        newLogs.shift();
      }
      return { logs: newLogs };
    }),
  clearLogs: () => set({ logs: [] }),

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
}));
