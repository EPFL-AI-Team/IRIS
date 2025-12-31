import { useCallback, useRef, useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import type { AnalysisResult, AnalysisLog } from "../types";

/**
 * Generate unique ID for analysis logs.
 */
function generateLogId(): string {
  return `log-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Hook to manage WebSocket connection for video analysis.
 * Connects to /ws/analysis and handles progress updates and results.
 */
export function useAnalysisWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const setAnalysisMode = useAppStore((state) => state.setAnalysisMode);
  const addAnalysisResult = useAppStore((state) => state.addAnalysisResult);
  const setAnalysisProgress = useAppStore((state) => state.setAnalysisProgress);
  const addLog = useAppStore((state) => state.addLog);
  const addAnalysisLog = useAppStore((state) => state.addAnalysisLog);
  const setAnalysisJobId = useAppStore((state) => state.setAnalysisJobId);

  const connect = useCallback(() => {
    // Clear any pending reconnection
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/analysis`;

    addLog("Connecting to analysis WebSocket...", "INFO");
    setAnalysisMode("running");

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      addLog("Analysis WebSocket connected", "INFO");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "result": {
            // Inference result received
            addAnalysisResult(data as AnalysisResult);
            addLog(
              `Result received for frames ${data.frame_range[0]}-${data.frame_range[1]}`,
              "INFO"
            );

            // Add to analysis logs (synced with video timeline)
            const videoTimeMs = data.timestamp_range_ms
              ? data.timestamp_range_ms[0]
              : null;

            // Parse result string to object if possible
            let inferenceResult: Record<string, unknown> | undefined;
            try {
              if (typeof data.result === "string") {
                inferenceResult = JSON.parse(data.result);
              } else {
                inferenceResult = data.result;
              }
            } catch {
              // Keep as string if not valid JSON
            }

            const analysisLog: AnalysisLog = {
              id: generateLogId(),
              timestamp: Date.now(),
              video_time_ms: videoTimeMs,
              type: "inference",
              message: `Frames ${data.frame_range[0]}-${data.frame_range[1]}`,
              inference_result: inferenceResult,
              inference_time_ms: data.inference_time ? data.inference_time * 1000 : undefined,
            };
            addAnalysisLog(analysisLog);
            break;
          }

          case "progress":
            // Progress update with enhanced fields
            setAnalysisProgress({
              current_frame: data.current_frame,
              total_frames: data.total_frames,
              progress_percent: data.progress_percent,
              position_ms: data.position_ms,
              current_chunk: data.current_chunk,
              total_chunks: data.total_chunks,
              estimated_time_remaining: data.estimated_time_remaining,
            });
            break;

          case "complete": {
            // Analysis complete
            setAnalysisMode("complete");
            setAnalysisProgress(null);
            addLog(
              `Analysis complete: ${data.total_frames} frames, ${data.total_results} results`,
              "INFO"
            );

            // Add completion log to analysis logs
            const completeLog: AnalysisLog = {
              id: generateLogId(),
              timestamp: Date.now(),
              video_time_ms: null,
              type: "system",
              message: `Analysis complete: ${data.total_frames} frames processed, ${data.total_results} results in ${data.duration_sec?.toFixed(1) || "?"}s`,
            };
            addAnalysisLog(completeLog);

            // Close the WebSocket
            ws.close();
            break;
          }

          case "error": {
            // Error occurred
            setAnalysisMode("error");
            addLog(`Analysis error: ${data.message}`, "ERROR");

            // Add error to analysis logs
            const errorLog: AnalysisLog = {
              id: generateLogId(),
              timestamp: Date.now(),
              video_time_ms: null,
              type: "error",
              message: data.message,
            };
            addAnalysisLog(errorLog);

            ws.close();
            break;
          }

          case "log": {
            // Log message from server
            addLog(data.message, data.level || "INFO");

            // Add system log to analysis logs
            const sysLog: AnalysisLog = {
              id: generateLogId(),
              timestamp: Date.now(),
              video_time_ms: null,
              type: "system",
              message: data.message,
            };
            addAnalysisLog(sysLog);
            break;
          }

          default:
            console.warn("Unknown analysis message type:", data.type);
        }
      } catch (error) {
        console.error("Failed to parse analysis WebSocket message:", error);
        addLog("Failed to parse analysis message", "ERROR");
      }
    };

    ws.onerror = (error) => {
      console.error("Analysis WebSocket error:", error);
      addLog("Analysis WebSocket error", "ERROR");
      setAnalysisMode("error");
    };

    ws.onclose = (event) => {
      if (event.code === 1000) {
        // Normal closure
        addLog("Analysis WebSocket closed", "INFO");
      } else {
        // Abnormal closure
        addLog(
          `Analysis WebSocket closed unexpectedly (code: ${event.code})`,
          "WARNING"
        );
        setAnalysisMode("error");
      }

      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [
    setAnalysisMode,
    addAnalysisResult,
    setAnalysisProgress,
    addLog,
    addAnalysisLog,
    setAnalysisJobId,
  ]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      addLog("Closing analysis WebSocket...", "INFO");
      wsRef.current.close();
      wsRef.current = null;
    }

    setAnalysisMode("idle");
    setAnalysisProgress(null);
  }, [setAnalysisMode, setAnalysisProgress, addLog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, []);

  return { connect, disconnect };
}
