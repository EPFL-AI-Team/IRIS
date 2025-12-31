import { useState, useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import { useAnalysisWebSocket } from "../hooks/useAnalysisWebSocket";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ReportModal } from "./ReportModal";
import { Play, Square, FileText, Clock } from "lucide-react";
import { toast } from "sonner";

/**
 * Format seconds into human-readable time string.
 */
function formatTime(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Component for controlling video analysis.
 * Includes start/stop buttons, progress display, and auto-report option.
 */
export function AnalysisControls() {
  const [reportModalOpen, setReportModalOpen] = useState(false);

  const analysisMode = useAppStore((state) => state.analysisMode);
  const segmentConfig = useAppStore((state) => state.segmentConfig);
  const progress = useAppStore((state) => state.analysisProgress);
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const selectedAnnotation = useAppStore(
    (state) => state.selectedAnnotationFile
  );
  const setAnalysisJobId = useAppStore((state) => state.setAnalysisJobId);
  const clearAnalysisResults = useAppStore(
    (state) => state.clearAnalysisResults
  );
  const clearAnalysisLogs = useAppStore((state) => state.clearAnalysisLogs);
  const addLog = useAppStore((state) => state.addLog);
  const analysisJobId = useAppStore((state) => state.analysisJobId);
  const autoGenerateReport = useAppStore((state) => state.autoGenerateReport);
  const setAutoGenerateReport = useAppStore(
    (state) => state.setAutoGenerateReport
  );

  const { connect, disconnect } = useAnalysisWebSocket();

  // Auto-open report modal when analysis completes and auto-generate is enabled
  useEffect(() => {
    if (analysisMode === "complete" && autoGenerateReport && analysisJobId) {
      setReportModalOpen(true);
    }
  }, [analysisMode, autoGenerateReport, analysisJobId]);

  const handleStart = async () => {
    if (!selectedVideo) {
      toast.error("Please select a video file");
      return;
    }

    try {
      // Call backend to start analysis with segment config
      const response = await fetch("/api/analysis/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_filename: selectedVideo,
          annotation_filename: selectedAnnotation,
          segment_time: segmentConfig.segmentTime,
          frames_per_segment: segmentConfig.framesPerSegment,
          overlap_frames: segmentConfig.overlapFrames,
        }),
      });

      const data = await response.json();

      if (response.ok && data.status === "ok") {
        setAnalysisJobId(data.job_id);
        clearAnalysisResults();
        clearAnalysisLogs();
        addLog(
          `Starting analysis: ${selectedVideo} (${data.total_frames} frames, ${data.total_chunks} chunks, ${data.annotation_count} annotations)`,
          "INFO"
        );
        toast.success("Analysis started");

        // Connect WebSocket to start streaming
        connect();
      } else {
        toast.error(data.message || "Failed to start analysis");
        addLog(`Failed to start analysis: ${data.message}`, "ERROR");
      }
    } catch (error) {
      console.error("Failed to start analysis:", error);
      toast.error("Failed to start analysis");
      addLog("Failed to start analysis", "ERROR");
    }
  };

  const handleStop = async () => {
    try {
      // Disconnect WebSocket first
      disconnect();

      // Call backend to stop analysis
      const response = await fetch("/api/analysis/stop", { method: "POST" });

      if (response.ok) {
        toast.info("Analysis stopped");
        addLog("Analysis stopped by user", "INFO");
      }
    } catch (error) {
      console.error("Failed to stop analysis:", error);
      toast.error("Failed to stop analysis");
    }
  };

  const isRunning = analysisMode === "running";
  const isComplete = analysisMode === "complete";
  const hasError = analysisMode === "error";

  return (
    <div className="space-y-4">
      {/* Controls Row */}
      <div className="flex gap-4 items-center flex-wrap">
        {/* Auto-generate report checkbox */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="auto-report"
            checked={autoGenerateReport}
            onCheckedChange={(checked: boolean | "indeterminate") =>
              setAutoGenerateReport(checked === true)
            }
            disabled={isRunning}
          />
          <Label
            htmlFor="auto-report"
            className="text-sm cursor-pointer select-none"
          >
            Auto-generate report
          </Label>
        </div>

        {!isRunning && (
          <Button onClick={handleStart} disabled={!selectedVideo}>
            <Play className="w-4 h-4 mr-1" />
            Run Analysis
          </Button>
        )}

        {isRunning && (
          <Button onClick={handleStop} variant="destructive">
            <Square className="w-4 h-4 mr-1" />
            Stop Analysis
          </Button>
        )}

        {isComplete && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-green-600 dark:text-green-400">
              Analysis complete
            </span>
            <Button onClick={handleStart} variant="outline" size="sm">
              Run Again
            </Button>
            <Button onClick={() => setReportModalOpen(true)} size="sm">
              <FileText className="w-4 h-4 mr-1" />
              Generate Report
            </Button>
          </div>
        )}

        {hasError && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-red-600 dark:text-red-400">
              Analysis failed
            </span>
            <Button onClick={handleStart} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        )}
      </div>

      {/* Progress Display */}
      {progress && isRunning && (
        <div className="space-y-2">
          <Progress value={progress.progress_percent} />
          <div className="flex justify-between text-xs text-muted-foreground">
            <div className="flex gap-4">
              <span>
                Frame {progress.current_frame} / {progress.total_frames}
              </span>
              {progress.current_chunk !== undefined &&
                progress.total_chunks !== undefined && (
                  <span>
                    Chunk {progress.current_chunk} / {progress.total_chunks}
                  </span>
                )}
            </div>
            <div className="flex items-center gap-2">
              {progress.estimated_time_remaining !== undefined &&
                progress.estimated_time_remaining > 0 && (
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />~
                    {formatTime(progress.estimated_time_remaining)} remaining
                  </span>
                )}
              <span>{progress.progress_percent.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Report Modal */}
      <ReportModal
        sessionId={analysisJobId || undefined}
        open={reportModalOpen}
        onOpenChange={setReportModalOpen}
      />
    </div>
  );
}
