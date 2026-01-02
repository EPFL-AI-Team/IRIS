import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";
import type { FrameData, WebSocketMessage, ResultItem, SessionAckMessage, SessionMetricsMessage } from "../types";

const RECONNECT_DELAY = 2000;

/**
 * Hook for streaming browser camera frames to the backend.
 * Connects to /ws/browser-stream which forwards frames to the inference server.
 */
export function useBrowserStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(0);
  const isConnectedRef = useRef(false);

  const setBrowserStreamConnection = useAppStore(
    (state) => state.setBrowserStreamConnection
  );
  const addLog = useAppStore((state) => state.addLog);
  const addResult = useAppStore((state) => state.addResult);
  const updateResult = useAppStore((state) => state.updateResult);
  const setSessionState = useAppStore((state) => state.setSessionState);
  const setSessionMetrics = useAppStore((state) => state.setSessionMetrics);
  const resetSessionState = useAppStore((state) => state.resetSessionState);

  const handleMessage = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (data: WebSocketMessage | SessionAckMessage | SessionMetricsMessage | any) => {
      switch (data.type) {
        case "session_ack": {
          // Session established with inference server
          const ack = data as SessionAckMessage;
          setSessionState({
            sessionId: ack.session_id,
            configured: true,
            mode: "live",
            config: {
              frames_per_segment: ack.config.frames_per_segment,
              overlap_frames: ack.config.overlap_frames,
            },
          });
          addLog(`Session established: ${ack.session_id}`, "INFO");
          break;
        }

        case "session_metrics": {
          // Live metrics update from inference server
          const metrics = data as SessionMetricsMessage;
          setSessionMetrics({
            elapsedSeconds: metrics.elapsed_seconds,
            segmentsProcessed: metrics.segments_processed,
            segmentsTotal: metrics.segments_total,
            queueDepth: metrics.queue_depth,
            processingRate: metrics.processing_rate,
            framesReceived: metrics.frames_received,
          });
          break;
        }

        case "batch_submitted": {
          const pendingResult: ResultItem = {
            id: `result-${data.job_id}`,
            job_id: data.job_id,
            timestamp: new Date(data.timestamp * 1000),
            status: "pending",
          };
          addResult(pendingResult);
          break;
        }

        case "log": {
          if (data.message?.includes("Starting inference:")) {
            updateResult(data.job_id, {
              status: "processing",
              pendingDetails: data.message,
            });
          }
          break;
        }

        case "result": {
          updateResult(data.job_id, {
            status: "completed",
            result: data.result,
            frames_processed: data.frames_processed,
            inference_time: data.inference_time,
            timestamp: new Date(data.timestamp * 1000),
          });
          break;
        }

        case "error":
          addLog(`Browser stream error: ${data.message}`, "ERROR");
          break;
      }
    },
    [addLog, addResult, updateResult, setSessionState, setSessionMetrics]
  );

  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setBrowserStreamConnection("connecting");
    addLog("Connecting to browser stream WebSocket...", "INFO");

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${protocol}//${window.location.host}/ws/browser-stream`
    );
    wsRef.current = ws;

    ws.onopen = () => {
      isConnectedRef.current = true;
      setBrowserStreamConnection("connected");

      // Wait for connection to be fully established before sending
      setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) {
          // Get segment config from store and send FIRST before any frames
          const { segmentConfig } = useAppStore.getState();
          const configMessage = {
            type: "segment_config",
            frames_per_segment: segmentConfig.framesPerSegment,
            overlap_frames: segmentConfig.overlapFrames,
          };
          ws.send(JSON.stringify(configMessage));
          addLog(
            `Sent segment config: frames=${segmentConfig.framesPerSegment}, overlap=${segmentConfig.overlapFrames}`,
            "INFO"
          );
        }
      }, 100);

      frameCountRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage;
        handleMessage(data);
      } catch (e) {
        console.error("Failed to parse browser stream message:", e);
      }
    };

    ws.onerror = () => {
      setBrowserStreamConnection("error");
      addLog("Browser stream WebSocket error", "ERROR");
    };

    ws.onclose = () => {
      isConnectedRef.current = false;
      setBrowserStreamConnection("disconnected");
      addLog("Browser stream WebSocket closed", "WARNING");

      // Auto-reconnect if we were connected
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connect();
      }, RECONNECT_DELAY);
    };
  }, [handleMessage, setBrowserStreamConnection, addLog]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    isConnectedRef.current = false;
    setBrowserStreamConnection("disconnected");
    resetSessionState();
  }, [setBrowserStreamConnection, resetSessionState]);

  /**
   * Send a frame to the backend for inference.
   * @param base64Jpeg Base64-encoded JPEG frame data
   * @param fps Target FPS for inference timing
   */
  const sendFrame = useCallback((base64Jpeg: string, fps: number = 5) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    const now = Date.now() / 1000;
    const measuredFps =
      lastFrameTimeRef.current > 0
        ? 1 / (now - lastFrameTimeRef.current)
        : 0;
    lastFrameTimeRef.current = now;

    const frameData: FrameData = {
      frame: base64Jpeg,
      frame_id: frameCountRef.current++,
      timestamp: now,
      fps,
      measured_fps: measuredFps,
    };

    try {
      wsRef.current.send(JSON.stringify(frameData));
      return true;
    } catch (e) {
      console.error("Failed to send frame:", e);
      return false;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    sendFrame,
    isConnected: isConnectedRef.current,
  };
}
