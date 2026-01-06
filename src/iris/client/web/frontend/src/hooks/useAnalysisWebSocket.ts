import { useCallback, useRef, useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import type { SessionAckMessage, SessionMetricsMessage, ResultItem } from "../types";
import { toast } from "sonner";

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
  const addAnalysisLog = useAppStore((state) => state.addAnalysisLog);
  const setSessionState = useAppStore((state) => state.setSessionState);
  const setAnalysisSessionMetrics = useAppStore((state) => state.setAnalysisSessionMetrics);
  const resetSessionState = useAppStore((state) => state.resetSessionState);

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

    addAnalysisLog("Connecting to analysis WebSocket...", "INFO");
    setAnalysisMode("running");

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      addAnalysisLog("Analysis WebSocket connected", "INFO");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "session_ack": {
            // Session established with inference server
            const ack = data as SessionAckMessage;
            setSessionState({
              sessionId: ack.session_id,
              configured: true,
              mode: "analysis",
              config: {
                frames_per_segment: ack.config.frames_per_segment,
                overlap_frames: ack.config.overlap_frames,
              },
            });
            addAnalysisLog(`Session established: ${ack.session_id}`, "INFO");
            break;
          }

          case "session_metrics": {
            // Live metrics update from inference server
            const metrics = data as SessionMetricsMessage;
            setAnalysisSessionMetrics({
              elapsedSeconds: metrics.elapsed_seconds,
              segmentsProcessed: metrics.segments_processed,
              segmentsTotal: metrics.segments_total,
              batchSize: metrics.batch_size,
            });

            // Update progress bar to show segment-based progress (concrete counts)
            if (metrics.segments_total && metrics.segments_total > 0) {
              const inferenceProgress =
                (metrics.segments_processed / metrics.segments_total) * 100;

              setAnalysisProgress({
                current_frame: metrics.segments_processed,  // Concrete segment count
                total_frames: metrics.segments_total,       // Total segments to process
                progress_percent: inferenceProgress,         // Percentage for progress bar
                position_ms: 0,
              });
            }
            break;
          }

          case "result": {
            // Inference result received - adapt to unified ResultItem
            const resultItem: ResultItem = {
              id: `result-${data.job_id}`,
              job_id: data.job_id as string,
              timestamp: new Date(data.timestamp_range_ms[0]),
              videoTimeMs: data.timestamp_range_ms[0],
              timestamp_range_ms: data.timestamp_range_ms, // Pass through range
              frame_range: data.frame_range, // Pass through frame range
              status: "completed",
              result: data.result as string,
              frames_processed: data.frames_processed as number,
              inference_time: data.inference_time as number,
            };
            addAnalysisResult(resultItem);
            addAnalysisLog(
              `Result received for frames ${data.frame_range[0]}-${data.frame_range[1]}`,
              "INFO"
            );
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
            addAnalysisLog(
              `Analysis complete: ${data.total_frames} frames, ${data.total_results} results`,
              "INFO"
            );

            // Show completion notification
            toast.success("Analysis Complete", {
              description: `Processed ${data.total_frames} frames with ${data.total_results} results in ${data.duration_sec?.toFixed(1) || "?"}s`,
            });

            // Close the WebSocket
            ws.close();
            break;
          }

          case "error": {
            // Error occurred
            setAnalysisMode("error");
            addAnalysisLog(`Analysis error: ${data.message}`, "ERROR");

            ws.close();
            break;
          }

          case "log": {
            // Log message from server
            addAnalysisLog(data.message, data.level || "INFO");
            break;
          }

          case "upload_complete": {
            // Video upload to server completed
            addAnalysisLog(
              `Upload complete: ${data.total_frames} frames uploaded in ${data.duration_sec?.toFixed(1) || "?"}s`,
              "INFO"
            );
            break;
          }

          default:
            console.warn("Unknown analysis message type:", data.type);
        }
      } catch (error) {
        console.error("Failed to parse analysis WebSocket message:", error);
        addAnalysisLog("Failed to parse analysis message", "ERROR");
      }
    };

    ws.onerror = (error) => {
      console.error("Analysis WebSocket error:", error);
      addAnalysisLog("Analysis WebSocket error", "ERROR");
      setAnalysisMode("error");
    };

    ws.onclose = (event) => {
      if (event.code === 1000) {
        // Normal closure
        addAnalysisLog("Analysis WebSocket closed", "INFO");
      } else {
        // Abnormal closure
        addAnalysisLog(
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
    addAnalysisLog,
    setSessionState,
    setAnalysisSessionMetrics,
  ]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      addAnalysisLog("Closing analysis WebSocket...", "INFO");
      wsRef.current.close();
      wsRef.current = null;
    }

    setAnalysisMode("idle");
    setAnalysisProgress(null);
    resetSessionState();
  }, [setAnalysisMode, setAnalysisProgress, addAnalysisLog, resetSessionState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
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
