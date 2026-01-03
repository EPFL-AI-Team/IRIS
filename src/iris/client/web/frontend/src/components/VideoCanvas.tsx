import { useRef, useEffect, useCallback } from "react";
import { useAppStore } from "../store/useAppStore";
import { Badge } from "@/components/ui/badge";

/**
 * VideoCanvas component for displaying camera preview.
 *
 * In the new architecture, preview frames come from the backend USB camera
 * via the unified WebSocket connection. The previewFrame state in the store
 * is set by useClientWebSocket hook.
 */
export function VideoCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const previewFrame = useAppStore((state) => state.previewFrame);
  const connectionStatus = useAppStore((state) => state.connectionStatus);
  const serverAlive = useAppStore((state) => state.serverAlive);

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
    } catch (error) {
      console.error("Error rendering frame:", error);
    }
  }, []);

  // Render preview frame when it updates
  useEffect(() => {
    if (previewFrame?.data) {
      renderFrame(previewFrame.data);
    }
  }, [previewFrame, renderFrame]);

  const getPreviewStatus = (): {
    label: string;
    detail?: string;
    variant: "default" | "secondary" | "destructive" | "outline";
  } => {
    if (connectionStatus !== "connected") {
      return {
        label: "Disconnected",
        detail: "Not connected to server",
        variant: "destructive",
      };
    }

    if (!serverAlive) {
      return {
        label: "Server Offline",
        detail: "Inference server not available",
        variant: "secondary",
      };
    }

    if (previewFrame) {
      return { label: "Server Camera", variant: "outline" };
    }

    return { label: "Waiting for preview...", variant: "secondary" };
  };

  const previewStatus = getPreviewStatus();

  return (
    <>
      <canvas
        ref={canvasRef}
        id="preview"
        className="flex-1 w-full min-h-0 object-contain bg-black"
      />
      <div
        id="preview-status"
        className="shrink-0 flex items-center gap-3 px-4 py-2 bg-background border-t border-border/50"
      >
        <Badge
          variant={previewStatus.variant}
          className="shrink-0 whitespace-nowrap"
        >
          {previewStatus.label}
        </Badge>

        {previewStatus.detail ? (
          <span className="text-xs text-muted-foreground truncate font-medium">
            {previewStatus.detail}
          </span>
        ) : null}
      </div>
    </>
  );
}
