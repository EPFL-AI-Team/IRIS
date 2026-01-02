import { useAppStore } from "../store/useAppStore";
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
  const sessionMetrics = useAppStore((state) => state.sessionMetrics);

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
      {sessionMetrics && (
        <>
          {/* Elapsed Time */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Time</span>
            <span className="font-mono font-medium text-foreground">
              {formatTime(sessionMetrics.elapsedSeconds)}
            </span>
          </div>

          {/* Segments */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Segments</span>
            <span className="font-mono font-medium text-foreground">
              {sessionMetrics.segmentsProcessed}
              {sessionMetrics.segmentsTotal !== null && (
                <span className="text-muted-foreground">
                  /{sessionMetrics.segmentsTotal}
                </span>
              )}
            </span>
          </div>

          {/* Queue Depth */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Queue</span>
            <span
              className={`font-mono font-medium ${
                sessionMetrics.queueDepth > 5
                  ? "text-yellow-500"
                  : "text-foreground"
              }`}
            >
              {sessionMetrics.queueDepth}
            </span>
          </div>

          {/* Processing Rate */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Rate</span>
            <span className="font-mono font-medium text-foreground">
              {sessionMetrics.processingRate.toFixed(1)}
              <span className="text-muted-foreground">/s</span>
            </span>
          </div>

          {/* Frames Received */}
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="text-[10px] uppercase tracking-wide">Frames</span>
            <span className="font-mono font-medium text-foreground">
              {sessionMetrics.framesReceived}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
