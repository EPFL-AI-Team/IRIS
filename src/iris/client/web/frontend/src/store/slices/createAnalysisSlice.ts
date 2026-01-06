import type { StateCreator } from "zustand";
import type {
  AnalysisMode,
  VideoInfo,
  AnnotationInfo,
  GroundTruthAnnotation,
  AnalysisProgress,
  ResultItem,
  AnalysisLog,
  LogLevel,
} from "../../types";

const MAX_ANALYSIS_RESULTS = 200;
const MAX_ANALYSIS_LOGS = 500;

export interface AnalysisSlice {
  // Analysis mode
  analysisMode: AnalysisMode;
  setAnalysisMode: (mode: AnalysisMode) => void;

  // Available datasets
  availableDatasets: {
    videos: VideoInfo[];
    annotations: AnnotationInfo[];
  } | null;
  setAvailableDatasets: (datasets: {
    videos: VideoInfo[];
    annotations: AnnotationInfo[];
  } | null) => void;

  // Selected files
  selectedVideoFile: string | null;
  setSelectedVideoFile: (filename: string | null) => void;

  selectedAnnotationFile: string | null;
  setSelectedAnnotationFile: (filename: string | null) => void;

  // Analysis job
  analysisJobId: string | null;
  setAnalysisJobId: (id: string | null) => void;

  analysisProgress: AnalysisProgress | null;
  setAnalysisProgress: (progress: AnalysisProgress | null) => void;

  // Analysis Results & Logs
  analysisResults: ResultItem[];
  addAnalysisResult: (result: ResultItem) => void;
  setAnalysisResults: (results: ResultItem[]) => void;
  clearAnalysisResults: () => void;

  analysisLogs: AnalysisLog[];
  addAnalysisLog: (log: string | Partial<AnalysisLog>, level?: LogLevel) => void;
  setAnalysisLogs: (logs: AnalysisLog[]) => void;
  clearAnalysisLogs: () => void;

  // Ground truth annotations
  groundTruthAnnotations: GroundTruthAnnotation[];
  setGroundTruthAnnotations: (annotations: GroundTruthAnnotation[]) => void;

  // Playback position
  currentPlaybackPosition: number; // milliseconds
  setCurrentPlaybackPosition: (ms: number) => void;

  // Auto-generate report
  autoGenerateReport: boolean;
  setAutoGenerateReport: (value: boolean) => void;
}

export const createAnalysisSlice: StateCreator<
  AnalysisSlice,
  [],
  [],
  AnalysisSlice
> = (set) => ({
  // Analysis mode
  analysisMode: "idle",
  setAnalysisMode: (mode) => set({ analysisMode: mode }),

  // Available datasets
  availableDatasets: null,
  setAvailableDatasets: (datasets) => set({ availableDatasets: datasets }),

  // Selected files
  selectedVideoFile: null,
  setSelectedVideoFile: (filename) => set({ selectedVideoFile: filename }),

  selectedAnnotationFile: null,
  setSelectedAnnotationFile: (filename) =>
    set({ selectedAnnotationFile: filename }),

  // Analysis job
  analysisJobId: null,
  setAnalysisJobId: (id) => set({ analysisJobId: id }),

  analysisProgress: null,
  setAnalysisProgress: (progress) => set({ analysisProgress: progress }),

  // Analysis Results & Logs
  analysisResults: [],
  addAnalysisResult: (result) =>
    set((state) => {
      const newResults = [...state.analysisResults, result];
      if (newResults.length > MAX_ANALYSIS_RESULTS) {
        newResults.shift();
      }
      return { analysisResults: newResults };
    }),
  setAnalysisResults: (results) => set({ analysisResults: results }),
  clearAnalysisResults: () => set({ analysisResults: [] }),

  analysisLogs: [],
  addAnalysisLog: (logInput, level = "INFO") =>
    set((state) => {
      let newLog: AnalysisLog;
      const timestamp = Date.now();
      const id = `log-analysis-${timestamp}-${Math.random().toString(36).slice(2, 9)}`;

      if (typeof logInput === "string") {
        // Create system/error log from string
        const type = level === "ERROR" ? "error" : "system";
        newLog = {
          id,
          timestamp,
          video_time_ms: null,
          type,
          message: logInput,
        };
      } else {
        // Use provided object, fill defaults
        newLog = {
          id,
          timestamp,
          video_time_ms: null,
          type: "system",
          message: "",
          ...logInput,
        };
      }

      const newLogs = [...state.analysisLogs, newLog];
      if (newLogs.length > MAX_ANALYSIS_LOGS) {
        newLogs.shift();
      }
      return { analysisLogs: newLogs };
    }),
  setAnalysisLogs: (logs) => set({ analysisLogs: logs }),
  clearAnalysisLogs: () => set({ analysisLogs: [] }),

  // Ground truth annotations
  groundTruthAnnotations: [],
  setGroundTruthAnnotations: (annotations) =>
    set({ groundTruthAnnotations: annotations }),

  // Playback position
  currentPlaybackPosition: 0,
  setCurrentPlaybackPosition: (ms) => set({ currentPlaybackPosition: ms }),

  // Auto-generate report
  autoGenerateReport: false,
  setAutoGenerateReport: (value) => set({ autoGenerateReport: value }),
});
