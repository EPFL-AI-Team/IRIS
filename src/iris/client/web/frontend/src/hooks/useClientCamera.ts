import { useEffect, useRef, useCallback, useState } from "react";
import { useAppStore } from "../store/useAppStore";

const DEFAULT_FPS = 10;
const JPEG_QUALITY = 0.8;

interface UseClientCameraOptions {
  fps?: number;
  width?: number;
  height?: number;
  onFrame?: (base64Jpeg: string) => void;
}

/**
 * Hook for accessing browser camera via getUserMedia.
 * Captures frames at specified FPS and provides them as base64 JPEG.
 */
export function useClientCamera(options: UseClientCameraOptions = {}) {
  const { fps = DEFAULT_FPS, width = 640, height = 480, onFrame } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setClientCameraPermission = useAppStore(
    (state) => state.setClientCameraPermission
  );
  const addLog = useAppStore((state) => state.addLog);

  // Initialize canvas for frame capture
  useEffect(() => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvasRef.current = canvas;
    ctxRef.current = canvas.getContext("2d");
  }, [width, height]);

  /**
   * Start the camera and begin capturing frames.
   */
  const start = useCallback(async () => {
    if (streamRef.current) {
      // Already running
      return true;
    }

    try {
      setError(null);
      addLog("Requesting camera access...", "INFO");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width, height },
      });

      streamRef.current = stream;
      setClientCameraPermission("granted");

      // Create hidden video element
      const video = document.createElement("video");
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      await video.play();
      videoRef.current = video;

      setIsActive(true);
      addLog("Browser camera initialized", "INFO");

      // Start frame capture loop
      const interval = 1000 / fps;
      captureIntervalRef.current = window.setInterval(() => {
        captureFrame();
      }, interval);

      return true;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to access camera";
      setError(message);
      setClientCameraPermission("denied");
      addLog(`Camera access failed: ${message}`, "ERROR");
      return false;
    }
  }, [width, height, fps, setClientCameraPermission, addLog]);

  /**
   * Stop the camera and cleanup resources.
   */
  const stop = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current = null;
    }

    setIsActive(false);
    addLog("Browser camera stopped", "INFO");
  }, [addLog]);

  /**
   * Capture a single frame from the video element.
   */
  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const ctx = ctxRef.current;
    const canvas = canvasRef.current;

    if (!video || !ctx || !canvas || video.readyState < 2) {
      return null;
    }

    // Update canvas size if video dimensions changed
    if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert to JPEG
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const base64 = dataUrl.split(",")[1];

    // Call onFrame callback if provided
    if (onFrame && base64) {
      onFrame(base64);
    }

    return base64;
  }, [onFrame]);

  /**
   * Get the current video element for direct rendering.
   */
  const getVideoElement = useCallback(() => {
    return videoRef.current;
  }, []);

  /**
   * Get the current frame as base64 JPEG (on-demand capture).
   */
  const getCurrentFrame = useCallback(() => {
    return captureFrame();
  }, [captureFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    start,
    stop,
    isActive,
    error,
    getVideoElement,
    getCurrentFrame,
  };
}
