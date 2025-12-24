import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";

const RECONNECT_DELAY = 2000;

/**
 * Hook for connecting to the server camera preview WebSocket.
 * Receives base64-encoded JPEG frames from /ws/preview.
 */
export function usePreviewWebSocket(
  onFrame: (base64Data: string) => void,
  enabled: boolean = true
) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const enabledRef = useRef(enabled);

  const setPreviewConnection = useAppStore(
    (state) => state.setPreviewConnection
  );
  const addLog = useAppStore((state) => state.addLog);

  // Keep enabled ref in sync
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  const connect = useCallback(() => {
    if (!enabledRef.current) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setPreviewConnection("connecting");
    addLog("Connecting to preview WebSocket...", "INFO");

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/preview`);
    wsRef.current = ws;

    ws.onopen = () => {
      setPreviewConnection("connected");
      addLog("Preview WebSocket connected", "INFO");
    };

    ws.onmessage = (event) => {
      // Frames are base64-encoded JPEG strings
      onFrame(event.data);
    };

    ws.onerror = () => {
      setPreviewConnection("error");
      addLog("Preview WebSocket error", "ERROR");
    };

    ws.onclose = () => {
      setPreviewConnection("disconnected");

      // Only reconnect if still enabled
      if (enabledRef.current) {
        addLog("Preview WebSocket closed, reconnecting...", "WARNING");
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, RECONNECT_DELAY);
      }
    };
  }, [onFrame, setPreviewConnection, addLog]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setPreviewConnection("disconnected");
  }, [setPreviewConnection]);

  // Connect when enabled, disconnect when disabled
  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    disconnect,
    reconnect: connect,
  };
}
