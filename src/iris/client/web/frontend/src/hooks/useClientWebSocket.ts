import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";
import type { ResultItem, SessionMetrics, AnalysisLog } from "../types";

// Singleton WebSocket instance shared across all hook calls
let sharedWs: WebSocket | null = null;
let lastFrameTime = 0;
let smoothedFps = 0; // Exponentially smoothed FPS
let lastServerAliveState: boolean | null = null; // Track server status to avoid repeated logs

/**
 * Simple WebSocket hook for unified frontend communication.
 *
 * Pattern from FastAPI docs:
 * 1. Connect on mount
 * 2. Handle messages
 * 3. Show "Disconnected" status on close
 * 4. User can click "Reconnect" button (no auto-reconnect)
 *
 * This replaces the previous separate hooks:
 * - useResultsWebSocket (for /ws/results)
 * - useBrowserStream (for /ws/browser-stream)
 * - useClientCamera (for browser camera)
 *
 * Note: Analysis-specific WebSocket is still handled by useAnalysisWebSocket
 * which connects to /ws/analysis endpoint.
 *
 * IMPORTANT: This hook uses a singleton WebSocket connection. Multiple components
 * can call this hook and they will all share the same connection.
 */
export function useClientWebSocket() {
  const wsRef = useRef<WebSocket | null>(sharedWs);
  const lastFrameTimeRef = useRef<number>(lastFrameTime);

  // Store setters
  const setConnectionStatus = useAppStore((state) => state.setConnectionStatus);
  const setPreviewFrame = useAppStore((state) => state.setPreviewFrame);
  const setServerAlive = useAppStore((state) => state.setServerAlive);
  const setSessionState = useAppStore((state) => state.setSessionState);
  const setSessionMetrics = useAppStore((state) => state.setSessionMetrics);
  const setIsStreaming = useAppStore((state) => state.setIsStreaming);
  const setFps = useAppStore((state) => state.setFps);
  const addResult = useAppStore((state) => state.addResult);
  const updateResult = useAppStore((state) => state.updateResult);
  const addLog = useAppStore((state) => state.addLog);
  const addAnalysisLog = useAppStore((state) => state.addAnalysisLog);

  // Store state
  const segmentConfig = useAppStore((state) => state.segmentConfig);
  const activeTab = useAppStore((state) => state.activeTab);

  const connect = useCallback(() => {
    // Check singleton first
    if (sharedWs?.readyState === WebSocket.OPEN) {
      wsRef.current = sharedWs;
      console.log("Reusing existing WebSocket connection");
      return;
    }

    setConnectionStatus("connecting");
    addLog("Connecting to server...", "INFO");

    // Always connect directly to client backend on localhost:8006
    const wsUrl = `ws://localhost:8006/ws/client`;
    console.log("Connecting to:", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    sharedWs = ws; // Update singleton

    ws.onopen = () => {
      setConnectionStatus("connected");
      console.log("WebSocket connected to:", wsUrl);
      addLog("Connected to server", "INFO");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onclose = (event) => {
      setConnectionStatus("disconnected");
      setServerAlive(false);
      setFps(0);
      lastFrameTimeRef.current = 0;
      lastFrameTime = 0; // Reset singleton
      smoothedFps = 0; // Reset smoothed FPS
      sharedWs = null; // Clear singleton
      addLog(`Disconnected from server (code: ${event.code})`, "WARNING");
      // NO AUTO-RECONNECT - user clicks Reconnect button
    };

    ws.onerror = () => {
      setConnectionStatus("error");
      addLog("WebSocket error", "ERROR");
    };
  }, [setConnectionStatus, setServerAlive, setFps, addLog]);

  const handleMessage = useCallback(
    (data: Record<string, unknown>) => {
      const msgType = data.type as string;

      switch (msgType) {
        case "session_info":
        case "session_ack": {
          const config = data.config as Record<string, unknown> | undefined;
          setSessionState({
            sessionId: data.session_id as string,
            config: config
              ? {
                  frames_per_segment: (config.frames_per_segment as number) || 8,
                  overlap_frames: (config.overlap_frames as number) || 4,
                }
              : null,
            configured: true,
          });
          addLog(`Session: ${data.session_id}`, "INFO");
          break;
        }

        case "preview_frame": {
          const timestamp = data.timestamp as number;
          setPreviewFrame(data.frame as string, timestamp);

          // Calculate FPS from preview frame timestamps with exponential smoothing
          const now = Date.now() / 1000; // Convert to seconds
          if (lastFrameTime > 0) {
            const elapsed = now - lastFrameTime;
            if (elapsed > 0 && elapsed < 1.0) { // Ignore gaps > 1 second
              const instantFps = 1 / elapsed;
              // Exponential smoothing: new = alpha * instant + (1-alpha) * old
              // alpha = 0.2 for heavy smoothing
              const alpha = 0.2;
              smoothedFps = alpha * instantFps + (1 - alpha) * smoothedFps;
              setFps(smoothedFps);
            }
          }
          lastFrameTime = now;
          lastFrameTimeRef.current = now;
          break;
        }

        case "result": {
          // Handle inference result - update existing pending card or add new
          console.log("Received result:", data);
          const result: ResultItem = {
            id: `result-${data.job_id}`,
            job_id: data.job_id as string,
            timestamp: new Date((data.timestamp as number) * 1000),
            status: "completed",
            result: data.result as string,
            frames_processed: data.frames_processed as number,
            inference_time: data.inference_time as number,
          };
          // Try to update existing pending result first
          updateResult(data.job_id as string, {
            status: "completed",
            result: data.result as string,
            frames_processed: data.frames_processed as number,
            inference_time: data.inference_time as number,
            timestamp: new Date((data.timestamp as number) * 1000),
          });
          // If no existing result, addResult handles dedup internally
          addResult(result);
          addLog(
            `Result: ${data.frames_processed as number} frames, ${((data.inference_time as number) || 0).toFixed(2)}s`,
            "INFO"
          );
          break;
        }

        case "batch_submitted": {
          // Create a pending result card
          const pendingResult: ResultItem = {
            id: `result-${data.job_id}`,
            job_id: data.job_id as string,
            timestamp: new Date((data.timestamp as number) * 1000),
            status: "pending",
          };
          addResult(pendingResult);
          break;
        }

        case "log": {
          // Handle log messages from server
          const level = (data.level as string) || "INFO";
          const message = data.message as string;
          const jobId = data.job_id as string | undefined;

          if (activeTab === "analysis") {
            const log: AnalysisLog = {
              id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
              timestamp: Date.now(),
              video_time_ms: null,
              type: level === "ERROR" ? "error" : "system",
              message: message,
            };
            addAnalysisLog(log);
          } else {
            addLog(message, level as "INFO" | "WARNING" | "ERROR" | "DEBUG");
          }

          // Update pending card with progress info if applicable
          if (jobId && message?.includes("Starting inference:")) {
            updateResult(jobId, {
              status: "processing",
              pendingDetails: message,
            });
          }
          break;
        }

        case "metrics":
        case "session_metrics": {
          const metrics: SessionMetrics = {
            elapsedSeconds: (data.elapsed_seconds as number) || 0,
            segmentsProcessed: (data.segments_processed as number) || 0,
            segmentsTotal: (data.segments_total as number) ?? null,
            queueDepth: (data.queue_depth as number) || 0,
            processingRate: (data.processing_rate as number) || 0,
            framesReceived: (data.frames_received as number) || 0,
          };
          setSessionMetrics(metrics);
          break;
        }

        case "server_status": {
          const alive = data.alive as boolean;
          setServerAlive(alive);
          // Log server status changes only when it actually changes
          if (lastServerAliveState !== null && lastServerAliveState !== alive) {
            if (alive) {
              addLog("Inference server is now online", "INFO");
            } else {
              addLog("Inference server is now offline", "WARNING");
            }
          }
          lastServerAliveState = alive;
          break;
        }

        case "streaming_started":
          setIsStreaming(true);
          addLog("Inference started", "INFO");
          break;

        case "streaming_stopped":
          setIsStreaming(false);
          addLog("Inference stopped", "INFO");
          break;

        case "queue_cleared":
          addLog(`Queue cleared: ${data.cleared} jobs removed`, "INFO");
          break;

        case "error":
          addLog(`Error: ${data.message}`, "ERROR");
          break;

        default:
          console.debug("Unknown message type:", msgType, data);
      }
    },
    [
      setSessionState,
      setPreviewFrame,
      setServerAlive,
      setSessionMetrics,
      setIsStreaming,
      setFps,
      addResult,
      updateResult,
      addLog,
      addAnalysisLog,
      activeTab,
    ]
  );

  const disconnect = useCallback(() => {
    if (sharedWs) {
      sharedWs.close();
      sharedWs = null;
    }
    wsRef.current = null;
    setConnectionStatus("disconnected");
    setFps(0);
    lastFrameTimeRef.current = 0;
    lastFrameTime = 0;
    smoothedFps = 0; // Reset smoothed FPS
  }, [setConnectionStatus, setFps]);

  // Send message helper
  const send = useCallback((message: Record<string, unknown>) => {
    if (sharedWs?.readyState === WebSocket.OPEN) {
      console.log("Sending WebSocket message:", message);
      sharedWs.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket not connected, cannot send:", message);
    }
  }, []);

  // Command helpers
  const startInference = useCallback(() => {
    console.log("startInference called, config:", segmentConfig);
    send({
      type: "start",
      config: {
        frames_per_segment: segmentConfig.framesPerSegment,
        overlap_frames: segmentConfig.overlapFrames,
      },
    });
  }, [send, segmentConfig]);

  const stopInference = useCallback(() => {
    send({ type: "stop" });
  }, [send]);

  const clearQueue = useCallback(() => {
    send({ type: "clear_queue" });
  }, [send]);

  const resetSession = useCallback(() => {
    send({ type: "reset_session" });
  }, [send]);

  const startAnalysis = useCallback(
    (
      videoPath: string,
      annotationPath: string | null,
      config: Record<string, unknown>
    ) => {
      send({
        type: "start_analysis",
        video_path: videoPath,
        annotation_path: annotationPath,
        config,
      });
    },
    [send]
  );

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connect,
    disconnect,
    startInference,
    stopInference,
    clearQueue,
    resetSession,
    startAnalysis,
    send,
  };
}
