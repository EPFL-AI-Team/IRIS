import { useRef, useEffect, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";
import { Badge } from "@/components/ui/badge";
import { usePreviewWebSocket } from "../hooks/usePreviewWebSocket";
import { useBrowserStream } from "../hooks/useBrowserStream";
import { useClientCamera } from "../hooks/useClientCamera";

/**
 * VideoCanvas component for displaying camera preview.
 * Supports both server camera (via WebSocket) and browser camera (via getUserMedia).
 */
export function VideoCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const lastFrameTimeRef = useRef(0);

  const cameraMode = useAppStore((state) => state.cameraMode);
  const isStreaming = useAppStore((state) => state.isStreaming);
  const clientVideoConfig = useAppStore((state) => state.clientVideoConfig);
  const clientCameraRequested = useAppStore(
    (state) => state.clientCameraRequested
  );
  const clearClientCameraRequest = useAppStore(
    (state) => state.clearClientCameraRequest
  );

  // Browser stream for sending frames to inference server
  const browserStream = useBrowserStream();

  /**
   * Render a base64-encoded JPEG frame to the canvas.
   */
  const renderFrame = useCallback(async (base64Data: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Initialize context if needed
    if (!ctxRef.current) {
      ctxRef.current = canvas.getContext("2d", {
        alpha: false,
        desynchronized: true,
      });
    }

    const ctx = ctxRef.current;
    if (!ctx) return;

    try {
      // Decode base64 to binary
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "image/jpeg" });

      // Create ImageBitmap for efficient rendering
      const imageBitmap = await createImageBitmap(blob);

      // Resize canvas if dimensions changed
      if (
        canvas.width !== imageBitmap.width ||
        canvas.height !== imageBitmap.height
      ) {
        canvas.width = imageBitmap.width;
        canvas.height = imageBitmap.height;
      }

      // Draw the frame
      ctx.drawImage(imageBitmap, 0, 0);

      // Clean up
      imageBitmap.close();

      // Track frame timing for debugging
      const now = performance.now();
      if (lastFrameTimeRef.current > 0) {
        // Could calculate actual render FPS here
      }
      lastFrameTimeRef.current = now;
    } catch (error) {
      console.error("Error rendering frame:", error);
    }
  }, []);

  // Frame handler for client camera
  const handleClientFrame = useCallback(
    (base64Jpeg: string) => {
      // Render to canvas
      renderFrame(base64Jpeg);

      // Send to inference server if streaming is active
      if (isStreaming) {
        browserStream.sendFrame(base64Jpeg, clientVideoConfig.capture_fps);
      }
    },
    [isStreaming, browserStream, renderFrame, clientVideoConfig.capture_fps]
  );

  // Client camera for capturing browser camera frames
  const clientCamera = useClientCamera({
    fps: clientVideoConfig.capture_fps,
    width: clientVideoConfig.width,
    height: clientVideoConfig.height,
    onFrame: handleClientFrame,
  });

  // Server preview WebSocket (only active in server camera mode)
  usePreviewWebSocket(renderFrame, cameraMode === "server");

  // Handle client camera start request from CameraSelector
  useEffect(() => {
    if (clientCameraRequested && cameraMode === "client") {
      clientCamera.start();
      clearClientCameraRequest();
    }
  }, [
    clientCameraRequested,
    cameraMode,
    clientCamera,
    clearClientCameraRequest,
  ]);

  // Handle camera mode changes
  useEffect(() => {
    if (cameraMode === "server") {
      // Stop client camera when switching to server mode
      clientCamera.stop();
    }
  }, [cameraMode, clientCamera]);

  // Connect/disconnect browser stream based on streaming state and camera mode
  useEffect(() => {
    if (cameraMode === "client" && isStreaming && clientCamera.isActive) {
      browserStream.connect();
    } else if (cameraMode !== "client" || !isStreaming) {
      browserStream.disconnect();
    }
  }, [cameraMode, isStreaming, clientCamera.isActive, browserStream]);

  const getPreviewStatus = (): {
    label: string;
    detail?: string;
    variant: "default" | "secondary" | "destructive" | "outline";
  } => {
    if (cameraMode === "client") {
      if (clientCamera.error) {
        return {
          label: "Camera Error",
          detail: clientCamera.error,
          variant: "destructive",
        };
      }
      if (clientCamera.isActive) {
        return { label: "Browser Camera", variant: "outline" };
      }
      return { label: "Camera Inactive", variant: "secondary" };
    }

    return { label: "Server Camera", variant: "outline" };
  };

  const previewStatus = getPreviewStatus();

  return (
    <>
      <canvas ref={canvasRef} id="preview"></canvas>
      <div id="preview-status" className="mt-2 flex items-center gap-2">
        <Badge variant={previewStatus.variant}>{previewStatus.label}</Badge>
        {previewStatus.detail ? (
          <span className="text-xs text-muted-foreground truncate">
            {previewStatus.detail}
          </span>
        ) : null}
      </div>
    </>
  );
}
