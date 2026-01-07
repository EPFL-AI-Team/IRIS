import { useRef, useEffect, useState } from "react";
import { useAppStore } from "../../store/useAppStore";
import { Loader2, AlertCircle } from "lucide-react";

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Get state from store
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const currentPlaybackPosition = useAppStore(
    (state) => state.currentPlaybackPosition
  ); // in ms
  const setCurrentPlaybackPosition = useAppStore(
    (state) => state.setCurrentPlaybackPosition
  );

  // 1. SYNC: Store -> Video
  // When the timeline is clicked (store changes), update the video current time.
  useEffect(() => {
    if (
      videoRef.current &&
      Math.abs(videoRef.current.currentTime * 1000 - currentPlaybackPosition) >
        200
    ) {
      // Only seek if difference is > 200ms to avoid stuttering during normal playback
      videoRef.current.currentTime = currentPlaybackPosition / 1000;
    }
  }, [currentPlaybackPosition]);

  // 2. SYNC: Video -> Store
  // As video plays, update the store (so the playhead moves).
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentPlaybackPosition(videoRef.current.currentTime * 1000);
    }
  };

  const handleLoadedData = () => {
    setIsLoading(false);
    setError(null);
  };

  const handleError = () => {
    setIsLoading(false);
    setError("Failed to load video. Please check if the file exists.");
  };

  const handleLoadStart = () => {
    setIsLoading(true);
    setError(null);
  };

  if (!selectedVideo) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground p-8 text-center">
        <div>
          <p className="text-sm font-medium">No video selected</p>
          <p className="text-xs mt-1 opacity-70">Select a dataset to begin</p>
        </div>
      </div>
    );
  }

  // Construct video URL (assuming your backend serves static files from /static/videos)
  const videoUrl = `/static/videos/${selectedVideo}`;

  return (
    // Added container with padding (p-4) and centering to fix "weird placement"
    <div className="relative w-full h-full flex items-center justify-center p-2">
      {/* Loading Spinner */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/50 backdrop-blur-sm">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/80">
          <div className="flex flex-col items-center gap-2 text-destructive">
            <AlertCircle className="w-8 h-8" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        </div>
      )}

      <video
        ref={videoRef}
        src={videoUrl}
        className="max-h-full max-w-full rounded-md shadow-sm outline-none"
        controls
        playsInline
        onTimeUpdate={handleTimeUpdate}
        onLoadedData={handleLoadedData}
        onError={handleError}
        onLoadStart={handleLoadStart}
      />
    </div>
  );
}
