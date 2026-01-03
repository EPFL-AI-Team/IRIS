import { useEffect, useRef, useState } from "react";
import { useAppStore } from "@/store/useAppStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  MessageSquare,
  Terminal,
  AlertCircle,
  Clock,
  Trash2,
  ChevronDown,
} from "lucide-react";
import type { AnalysisLog } from "@/types";
import { formatResultAsNaturalLanguage } from "@/utils/formatResult";

/**
 * Format timestamp as relative video time.
 */
function formatVideoTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/**
 * Get icon and color for log type.
 */
function getLogTypeStyle(type: AnalysisLog["type"]) {
  switch (type) {
    case "inference":
      return {
        icon: MessageSquare,
        color: "text-blue-500",
        bg: "bg-blue-500/10",
      };
    case "system":
      return {
        icon: Terminal,
        color: "text-muted-foreground",
        bg: "bg-muted/50",
      };
    case "error":
      return {
        icon: AlertCircle,
        color: "text-red-500",
        bg: "bg-red-500/10",
      };
  }
}

interface LogEntryProps {
  log: AnalysisLog;
  isActive: boolean;
  onSeek: (timeMs: number) => void;
}

function LogEntry({ log, isActive, onSeek }: LogEntryProps) {
  const style = getLogTypeStyle(log.type);
  const Icon = style.icon;

  return (
    <div
      className={`p-2 rounded-md cursor-pointer transition-colors ${style.bg} ${
        isActive ? "ring-2 ring-primary" : "hover:ring-1 hover:ring-muted-foreground/30"
      }`}
      onClick={() => log.video_time_ms !== null && onSeek(log.video_time_ms)}
    >
      <div className="flex items-start gap-2">
        <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${style.color}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            {log.video_time_ms !== null && (
              <span className="font-mono">
                {formatVideoTime(log.video_time_ms)}
              </span>
            )}
            {log.inference_time_ms !== undefined && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {log.inference_time_ms.toFixed(0)}ms
              </span>
            )}
          </div>
          <div className="text-sm wrap-break-word">
            {log.type === "inference" && log.inference_result ? (
              <div className="space-y-1">
                {/* Primary display: Natural language */}
                <div className="text-sm font-medium text-foreground leading-relaxed">
                  {formatResultAsNaturalLanguage(log.inference_result as Record<string, unknown>)}
                </div>

                {/* Secondary: Collapsed JSON details */}
                <details className="text-xs group">
                  <summary className="text-muted-foreground cursor-pointer hover:text-foreground transition-colors list-none inline-flex items-center gap-1 py-0.5">
                    <span className="inline-block transition-transform group-open:rotate-90 text-[10px]">▶</span>
                    <span className="text-[10px]">JSON</span>
                  </summary>
                  <pre className="text-[10px] font-mono whitespace-pre-wrap overflow-hidden mt-1 p-2 bg-muted/30 rounded border border-border/50">
                    {JSON.stringify(log.inference_result, null, 2)}
                  </pre>
                </details>
              </div>
            ) : (
              <span>{log.message}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Panel displaying analysis logs synced with video playback.
 * Shows model outputs and system logs with time indicators.
 */
export function LogPanel() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const analysisLogs = useAppStore((state) => state.analysisLogs);
  const clearAnalysisLogs = useAppStore((state) => state.clearAnalysisLogs);
  const currentPlaybackPosition = useAppStore(
    (state) => state.currentPlaybackPosition
  );
  const setCurrentPlaybackPosition = useAppStore(
    (state) => state.setCurrentPlaybackPosition
  );
  const segmentConfig = useAppStore((state) => state.segmentConfig);
  const analysisMode = useAppStore((state) => state.analysisMode);

  // Find the active log based on current playback position
  const activeLogIndex = analysisLogs.findIndex((log) => {
    if (log.video_time_ms === null) return false;
    const threshold = segmentConfig.segmentTime * 1000; // Segment duration as threshold
    return (
      log.video_time_ms <= currentPlaybackPosition &&
      log.video_time_ms + threshold > currentPlaybackPosition
    );
  });

  // Auto-scroll to active log or bottom during analysis
  useEffect(() => {
    if (!autoScroll || !scrollRef.current) return;

    if (analysisMode === "running") {
      // During analysis, scroll to bottom to show latest
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    } else if (activeLogIndex >= 0) {
      // During playback, scroll to active log
      const logElements = scrollRef.current.querySelectorAll("[data-log-entry]");
      const activeElement = logElements[activeLogIndex];
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [analysisLogs.length, activeLogIndex, autoScroll, analysisMode]);

  const handleSeek = (timeMs: number) => {
    setCurrentPlaybackPosition(timeMs);
  };

  const handleScroll = () => {
    if (!scrollRef.current) return;
    // Disable auto-scroll if user scrolls up
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(isAtBottom);
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="py-3 px-4 shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Analysis Logs</CardTitle>
          <div className="flex items-center gap-2">
            {!autoScroll && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => {
                  setAutoScroll(true);
                  if (scrollRef.current) {
                    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                  }
                }}
              >
                <ChevronDown className="w-3 h-3 mr-1" />
                Latest
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={clearAnalysisLogs}
              disabled={analysisLogs.length === 0}
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0 overflow-hidden">
        <div
          ref={scrollRef}
          className="h-full px-4 pb-4 overflow-y-auto"
          onScroll={handleScroll}
        >
          {analysisLogs.length === 0 ? (
            <div className="text-sm text-muted-foreground text-center py-8">
              No logs yet. Start an analysis to see results.
            </div>
          ) : (
            <div className="space-y-2">
              {analysisLogs.map((log, index) => (
                <div key={log.id} data-log-entry>
                  <LogEntry
                    log={log}
                    isActive={index === activeLogIndex}
                    onSeek={handleSeek}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
