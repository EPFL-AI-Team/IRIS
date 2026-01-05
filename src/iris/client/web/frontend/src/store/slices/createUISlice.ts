import type { StateCreator } from "zustand";
import type {
  ActiveTab,
  ConnectionStatus,
  LogEntry,
  LogLevel,
  ResultItem,
  SegmentConfig,
  SessionState,
  SessionMetrics,
} from "../../types";

const MAX_LOG_ENTRIES = 100;
const MAX_RESULTS = 50;

export interface UISlice {
  // Active tab
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;

  // Unified WebSocket connection status
  connectionStatus: ConnectionStatus;
  setConnectionStatus: (status: ConnectionStatus) => void;

  // Inference server health status
  serverAlive: boolean;
  setServerAlive: (alive: boolean) => void;

  // Live view results
  results: ResultItem[];
  addResult: (result: ResultItem) => void;
  updateResult: (jobId: string, updates: Partial<ResultItem>) => void;
  clearResults: () => void;
  setResults: (results: ResultItem[]) => void;

  // System logs
  logs: LogEntry[];
  addLog: (message: string, level?: LogLevel) => void;
  clearLogs: () => void;
  setLogs: (logs: LogEntry[]) => void;

  // Segment configuration (shared between live and analysis)
  segmentConfig: SegmentConfig;
  setSegmentConfig: (config: SegmentConfig) => void;

  // Session state (shared between Live View and Analysis View)
  sessionState: SessionState;
  setSessionState: (state: Partial<SessionState>) => void;
  resetSessionState: () => void;

  sessionMetrics: SessionMetrics | null;
  setSessionMetrics: (metrics: SessionMetrics | null) => void;

  // Report generation status
  reportStatus: "idle" | "generating" | "ready" | "error";
  setReportStatus: (status: "idle" | "generating" | "ready" | "error") => void;

  reportContent: string | null;
  setReportContent: (content: string | null) => void;

  reportError: string | null;
  setReportError: (error: string | null) => void;
}

export const createUISlice: StateCreator<UISlice, [], [], UISlice> = (set) => ({
  // Active tab - default to live view
  activeTab: "live",
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Unified WebSocket connection status
  connectionStatus: "disconnected",
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  // Inference server health status
  serverAlive: false,
  setServerAlive: (alive) => set({ serverAlive: alive }),

  // Live view results
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
  setResults: (results) => set({ results }),

  // System logs
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
  setLogs: (logs) => set({ logs }),

  // Segment configuration - defaults: T=1s, s=2 frames, overlap=0 → FPS=2
  segmentConfig: {
    segmentTime: 1.0,
    framesPerSegment: 2,
    overlapFrames: 0,
  },
  setSegmentConfig: (config) => set({ segmentConfig: config }),

  // Session state
  sessionState: {
    sessionId: null,
    configured: false,
    mode: null,
    config: null,
  },
  setSessionState: (newState) =>
    set((state) => ({
      sessionState: { ...state.sessionState, ...newState },
    })),
  resetSessionState: () =>
    set({
      sessionState: {
        sessionId: null,
        configured: false,
        mode: null,
        config: null,
      },
      sessionMetrics: null,
      results: [],
      logs: [],
      // Clear analysis state as well since they are in the same store
      analysisResults: [],
      analysisLogs: [],
      analysisJobId: null,
      analysisProgress: null,
      analysisMode: "idle",
      // Reset report state
      reportStatus: "idle",
      reportContent: null,
      reportError: null,
    } as unknown as UISlice),

  // Session metrics
  sessionMetrics: null,
  setSessionMetrics: (metrics) => set({ sessionMetrics: metrics }),

  // Report generation status
  reportStatus: "idle",
  setReportStatus: (status) => set({ reportStatus: status }),

  reportContent: null,
  setReportContent: (content) => set({ reportContent: content }),

  reportError: null,
  setReportError: (error) => set({ reportError: error }),
});
