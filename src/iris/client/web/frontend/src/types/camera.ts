// Camera-related types

export interface ServerConfig {
  host: string;
  port: number;
  endpoint: string;
}

export interface ClientVideoConfig {
  width: number;
  height: number;
  capture_fps: number;
  jpeg_quality: number;
  camera_index: number;
}

export interface CameraInfo {
  index: number;
  name: string;
  resolution: string;
}

export interface PreviewFrame {
  data: string; // base64 JPEG
  timestamp: number;
}

export interface FrameData {
  frame: string; // base64 encoded JPEG
  frame_id: number;
  timestamp: number;
  fps: number;
  measured_fps?: number;
}
