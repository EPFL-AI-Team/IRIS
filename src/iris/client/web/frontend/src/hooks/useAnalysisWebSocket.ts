import { useCallback, useRef, useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import type { AnalysisResult } from "../types";

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
          case "result":
            // Inference result received
            addAnalysisResult(data as AnalysisResult);
            addLog(
              `Result received for frames ${data.frame_range[0]}-${data.frame_range[1]}`,
              "INFO"
            );
            break;

          case "progress":
            // Progress update
            setAnalysisProgress({
              current_frame: data.current_frame,
              total_frames: data.total_frames,
              progress_percent: data.progress_percent,
              position_ms: data.position_ms,
            });
            break;

          case "complete":
            // Analysis complete
            setAnalysisMode("complete");
            setAnalysisProgress(null);
            addLog(
              `Analysis complete: ${data.total_frames} frames, ${data.total_results} results`,
              "INFO"
            );
            // Close the WebSocket
            ws.close();
            break;

          case "error":
            // Error occurred
            setAnalysisMode("error");
            addLog(`Analysis error: ${data.message}`, "ERROR");
            ws.close();
            break;

          case "log":
            // Log message from server
            addLog(data.message, data.level || "INFO");
            break;

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
