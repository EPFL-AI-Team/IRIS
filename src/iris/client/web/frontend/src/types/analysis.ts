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

/**
 * Segment configuration for video analysis
 * FPS is derived: FPS = framesPerSegment / segmentTime
 */
export interface SegmentConfig {
  segmentTime: number; // T - segment duration in seconds (e.g., 1.0)
  framesPerSegment: number; // s - frames sampled per segment (e.g., 8)
  overlapFrames: number; // overlap between segments (e.g., 4)
}

/**
 * Log entry for analysis (model outputs + system logs)
 */
export interface AnalysisLog {
  id: string;
  timestamp: number; // Unix timestamp ms
  video_time_ms: number | null; // Video position when log occurred
  type: "inference" | "system" | "error";
  message: string;
  inference_result?: Record<string, unknown>; // Parsed inference result if type is "inference"
  inference_time_ms?: number; // Duration of inference
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
  // Enhanced progress fields
  current_chunk?: number;
  total_chunks?: number;
  estimated_time_remaining?: number; // seconds
  processing_rate?: number; // chunks per second
}

export type AnalysisMode = "idle" | "running" | "paused" | "complete" | "error";
