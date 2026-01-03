import type { StateCreator } from "zustand";
import type {
  AnalysisMode,
  VideoInfo,
  AnnotationInfo,
  GroundTruthAnnotation,
  AnalysisResult,
  AnalysisProgress,
  AnalysisLog,
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

  // Analysis results
  analysisResults: AnalysisResult[];
  addAnalysisResult: (result: AnalysisResult) => void;
  clearAnalysisResults: () => void;

  // Ground truth annotations
  groundTruthAnnotations: GroundTruthAnnotation[];
  setGroundTruthAnnotations: (annotations: GroundTruthAnnotation[]) => void;

  // Playback position
  currentPlaybackPosition: number; // milliseconds
  setCurrentPlaybackPosition: (ms: number) => void;

  // Analysis logs
  analysisLogs: AnalysisLog[];
  addAnalysisLog: (log: AnalysisLog) => void;
  clearAnalysisLogs: () => void;

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

  // Analysis results
  analysisResults: [],
  addAnalysisResult: (result) =>
    set((state) => {
      const newResults = [...state.analysisResults, result];
      if (newResults.length > MAX_ANALYSIS_RESULTS) {
        newResults.shift();
      }
      return { analysisResults: newResults };
    }),
  clearAnalysisResults: () => set({ analysisResults: [] }),

  // Ground truth annotations
  groundTruthAnnotations: [],
  setGroundTruthAnnotations: (annotations) =>
    set({ groundTruthAnnotations: annotations }),

  // Playback position
  currentPlaybackPosition: 0,
  setCurrentPlaybackPosition: (ms) => set({ currentPlaybackPosition: ms }),

  // Analysis logs
  analysisLogs: [],
  addAnalysisLog: (log) =>
    set((state) => {
      const newLogs = [...state.analysisLogs, log];
      if (newLogs.length > MAX_ANALYSIS_LOGS) {
        newLogs.shift();
      }
      return { analysisLogs: newLogs };
    }),
  clearAnalysisLogs: () => set({ analysisLogs: [] }),

  // Auto-generate report
  autoGenerateReport: false,
  setAutoGenerateReport: (value) => set({ autoGenerateReport: value }),
});
