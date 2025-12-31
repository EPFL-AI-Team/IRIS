import { useEffect, useRef } from "react";
import { useAppStore } from "../store/useAppStore";

/**
 * HTML5 video player component for analysis playback.
 * Syncs playback position with the store and responds to timeline seeks.
 */
export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const currentPlaybackPosition = useAppStore(
    (state) => state.currentPlaybackPosition
  );
  const setCurrentPlaybackPosition = useAppStore(
    (state) => state.setCurrentPlaybackPosition
  );

  // Handle timeupdate event to sync position with store
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentPlaybackPosition(video.currentTime * 1000);
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => video.removeEventListener("timeupdate", handleTimeUpdate);
  }, [setCurrentPlaybackPosition]);

  // Seek to position when currentPlaybackPosition changes externally
  // (e.g., from timeline click)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const targetTimeSec = currentPlaybackPosition / 1000;
    const currentTimeSec = video.currentTime;

    // Only seek if the difference is significant (> 0.5 seconds)
    // to avoid feedback loops with timeupdate
    if (Math.abs(targetTimeSec - currentTimeSec) > 0.5) {
      video.currentTime = targetTimeSec;
    }
  }, [currentPlaybackPosition]);

  if (!selectedVideo) {
    return (
      <div className="aspect-video bg-black rounded-lg flex items-center justify-center">
        <p className="text-muted-foreground">No video selected</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        src={`/static/videos/${selectedVideo}`}
        controls
        className="w-full aspect-video bg-black rounded-lg"
      />
    </div>
  );
}
