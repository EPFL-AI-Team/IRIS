import { useAppStore } from "@/store/useAppStore";
import { Badge } from "@/components/ui/badge";

/**
 * Format seconds to MM:SS display.
 */
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Compact session metrics display component.
 * Shows real-time session info from inference server.
 */
export function SessionMetrics() {
  const sessionState = useAppStore((state) => state.sessionState);
  const liveSessionMetrics = useAppStore((state) => state.liveSessionMetrics);

  // Don't render if no session configured
  if (!sessionState.configured || !sessionState.sessionId) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-3 text-xs">
      {/* Session ID Badge */}
      <Badge variant="outline" className="font-mono text-[10px]">
        {sessionState.sessionId}
      </Badge>

      {/* Mode Badge */}
      <Badge variant={sessionState.mode === "live" ? "default" : "secondary"}>
        {sessionState.mode === "live" ? "Live" : "Analysis"}
      </Badge>

      {/* Metrics - only show if available */}
      {liveSessionMetrics && (
        <>
          {/* Elapsed Time */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Time</span>
            <span className="font-mono font-medium text-foreground">
              {formatTime(liveSessionMetrics.elapsedSeconds)}
            </span>
          </div>

          {/* Segments */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Segments</span>
            <span className="font-mono font-medium text-foreground">
              {liveSessionMetrics.segmentsProcessed}
              {liveSessionMetrics.segmentsTotal !== null && (
                <span className="text-muted-foreground">
                  /{liveSessionMetrics.segmentsTotal}
                </span>
              )}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
