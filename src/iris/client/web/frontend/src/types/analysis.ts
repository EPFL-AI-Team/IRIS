/**
 * Types for Analysis & Benchmark feature
 */

export interface VideoInfo {
  filename: string;
  path: string;
  size_mb: number;
  duration_sec: number;
  resolution: string;
  fps: number;
  frame_count: number;
}

export interface AnnotationInfo {
  filename: string;
  path: string;
  size_kb: number;
  line_count: number;
}

export interface GroundTruthAnnotation {
  start_ms: number;
  end_ms: number;
  start_sec: number;
  end_sec: number;
  action: string;
  tool: string;
  target: string;
  context: string;
}

export interface AnalysisResult {
  type: "result";
  job_id: string;
  status: "completed" | "pending";
  result: string; // JSON string
  frames_processed: number;
  inference_time: number;
  frame_range: [number, number];
  timestamp_range_ms: [number, number];
}

export interface AnalysisProgress {
  current_frame: number;
  total_frames: number;
  progress_percent: number;
  position_ms: number;
}

export type AnalysisMode = "idle" | "running" | "paused" | "complete" | "error";
