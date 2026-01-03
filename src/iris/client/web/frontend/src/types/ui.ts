// UI-specific types

export type LogLevel = "INFO" | "WARNING" | "ERROR" | "DEBUG";

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  message: string;
}

export interface ResultItem {
  id: string;
  job_id: string;
  timestamp: Date;
  status: "pending" | "processing" | "completed";
  result?: string;
  frames_processed?: number;
  inference_time?: number; // Time spent in inference (from server)
  latency?: number; // Round-trip time from submission to result (ms)
  submittedAt?: Date; // When the batch was submitted
  pendingDetails?: string;
}

export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

export type ActiveTab = "live" | "analysis";

export interface ApiResponse<T = unknown> {
  status: "ok" | "error";
  message?: string;
  data?: T;
}
