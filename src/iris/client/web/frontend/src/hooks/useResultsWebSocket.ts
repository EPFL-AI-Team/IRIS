import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";
import type {
  WebSocketMessage,
  ResultItem,
  ConnectionStatus,
} from "../types";

const RECONNECT_DELAY = 2000;

/**
 * Hook for connecting to the results WebSocket.
 * Receives inference results and status updates from /ws/results.
 */
export function useResultsWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const connectRef = useRef<(() => void) | null>(null);

  const setResultsConnection = useAppStore(
    (state) => state.setResultsConnection
  );
  const addLog = useAppStore((state) => state.addLog);
  const addResult = useAppStore((state) => state.addResult);
  const updateResult = useAppStore((state) => state.updateResult);
  const setIsStreaming = useAppStore((state) => state.setIsStreaming);
  const setIsCameraActive = useAppStore((state) => state.setIsCameraActive);
  const setFps = useAppStore((state) => state.setFps);
  const setStreamingServerStatus = useAppStore(
    (state) => state.setStreamingServerStatus
  );
  const setServerConfig = useAppStore((state) => state.setServerConfig);
  const setSSHTunnelConfig = useAppStore((state) => state.setSSHTunnelConfig);
  const setClientVideoConfig = useAppStore((state) => state.setClientVideoConfig);
  const resultsReconnectToken = useAppStore((state) => state.resultsReconnectToken);

  const handleMessage = useCallback(
    (data: WebSocketMessage) => {
      switch (data.type) {
        case "keepalive":
          // Ignore keepalive pings
          break;

        case "status_update":
          // Update all status fields from backend
          setIsCameraActive(data.camera_active);
          setIsStreaming(data.streaming_active);
          setFps(data.fps);
          setStreamingServerStatus(
            data.streaming_server_status as ConnectionStatus
          );
          if (data.config?.server) {
            setServerConfig(data.config.server);
          }
          if (data.config?.video) {
            setClientVideoConfig(data.config.video);
          }
          if (data.config?.ssh_tunnel) {
            setSSHTunnelConfig(data.config.ssh_tunnel);
          }
          break;

        case "batch_submitted": {
          // Create a pending result card
          const pendingResult: ResultItem = {
            id: `result-${data.job_id}`,
            job_id: data.job_id,
            timestamp: new Date(data.timestamp * 1000),
            status: "pending",
          };
          addResult(pendingResult);
          addLog(`Batch submitted: ${data.job_id}`, "INFO");
          break;
        }

        case "log": {
          // Update pending card with progress info
          if (data.message?.includes("Starting inference:")) {
            updateResult(data.job_id, {
              status: "processing",
              pendingDetails: data.message,
            });
          }
          break;
        }

        case "result": {
          // Update the pending card with final result
          const completedResult: Partial<ResultItem> = {
            status: "completed",
            result: data.result,
            frames_processed: data.frames_processed,
            inference_time: data.inference_time,
            timestamp: new Date(data.timestamp * 1000),
          };
          updateResult(data.job_id, completedResult);
          addLog(
            `Result received: ${data.job_id} (${data.frames_processed} frames, ${data.inference_time.toFixed(3)}s)`,
            "INFO"
          );
          break;
        }

        case "error":
          addLog(`Server error: ${data.message}`, "ERROR");
          break;
      }
    },
    [
      addLog,
      addResult,
      updateResult,
      setIsStreaming,
      setIsCameraActive,
      setFps,
      setStreamingServerStatus,
      setServerConfig,
      setClientVideoConfig,
      setSSHTunnelConfig,
    ]
  );

  const connect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setResultsConnection("connecting");
    addLog("Connecting to results WebSocket...", "INFO");

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/results`);
    wsRef.current = ws;

    ws.onopen = () => {
      setResultsConnection("connected");
      addLog("Results WebSocket connected", "INFO");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage;
        handleMessage(data);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = () => {
      setResultsConnection("error");
      addLog("Results WebSocket error", "ERROR");
    };

    ws.onclose = (event) => {
      setResultsConnection("disconnected");
      addLog(`Results WebSocket closed: code=${event.code}`, "WARNING");

      // Only auto-reconnect on abnormal closure (not clean close)
      if (event.code !== 1000 && event.code !== 1001) {
        addLog("Unexpected close, reconnecting...", "WARNING");
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectRef.current?.();
        }, RECONNECT_DELAY);
      }
    };
  }, [handleMessage, setResultsConnection, addLog]);

  // Keep a stable reference to the latest connect() to avoid TDZ/self-reference issues.
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setResultsConnection("disconnected");
  }, [setResultsConnection]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Manual reconnect requests from UI
  useEffect(() => {
    if (resultsReconnectToken <= 0) return;
    disconnect();
    connect();
  }, [resultsReconnectToken, connect, disconnect]);

  return {
    disconnect,
    reconnect: connect,
  };
}
