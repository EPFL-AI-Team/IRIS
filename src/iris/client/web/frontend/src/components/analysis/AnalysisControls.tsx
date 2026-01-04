import { useState, useEffect } from "react";
import { useAppStore } from "../../store/useAppStore";
import { useAnalysisWebSocket } from "../../hooks/useAnalysisWebSocket";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ReportModal } from "../ReportModal";
import { Play, Square, FileText } from "lucide-react";
import { toast } from "sonner";

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
  const analysisJobId = useAppStore((state) => state.analysisJobId);
  const autoGenerateReport = useAppStore((state) => state.autoGenerateReport);
  const setAutoGenerateReport = useAppStore(
    (state) => state.setAutoGenerateReport
  );
  const sessionMetrics = useAppStore((state) => state.sessionMetrics);
  const { connect, disconnect } = useAnalysisWebSocket();

  useEffect(() => {
    if (analysisMode === "complete" && autoGenerateReport && analysisJobId) {
      // Wrap in setTimeout to defer the state update prevents the cascading render warning
      const timer = setTimeout(() => {
        setReportModalOpen(true);
      }, 0);

      return () => clearTimeout(timer);
    }
  }, [analysisMode, autoGenerateReport, analysisJobId]);

  const handleStart = async () => {
    if (!selectedVideo) {
      toast.error("Please select a video file");
      return;
    }
    try {
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
        toast.success("Analysis started");
        connect();
      } else {
        toast.error(data.message || "Failed to start analysis");
      }
    } catch (error) {
      console.error("Failed to start analysis:", error);
      toast.error("Failed to start analysis");
    }
  };

  const handleStop = async () => {
    try {
      disconnect();
      await fetch("/api/analysis/stop", { method: "POST" });
      toast.info("Analysis stopped");
    } catch (error) {
      console.error("Failed to stop analysis:", error);
    }
  };

  const isRunning = analysisMode === "running";
  const isComplete = analysisMode === "complete";

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Row 1: Action Buttons */}
      <div className="flex items-center gap-2">
        {!isRunning ? (
          <Button
            onClick={handleStart}
            disabled={!selectedVideo}
            size="sm"
            className="flex-1 h-8 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            <Play className="w-3.5 h-3.5 mr-1.5 fill-current" />
            {isComplete ? "Run Again" : "Start Analysis"}
          </Button>
        ) : (
          <Button
            onClick={handleStop}
            variant="destructive"
            size="sm"
            className="flex-1 h-8 text-xs font-semibold"
          >
            <Square className="w-3.5 h-3.5 mr-1.5 fill-current" />
            Stop
          </Button>
        )}

        {isComplete && (
          <Button
            onClick={() => setReportModalOpen(true)}
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0"
            title="Generate Report"
          >
            <FileText className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Row 2: Options & Progress */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Checkbox
            id="auto-report"
            checked={autoGenerateReport}
            onCheckedChange={(c) => setAutoGenerateReport(c === true)}
            disabled={isRunning}
            className="h-3.5 w-3.5"
          />
          <Label
            htmlFor="auto-report"
            className="text-xs cursor-pointer text-muted-foreground"
          >
            Auto-generate report on finish
          </Label>
        </div>

        {progress && isRunning && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-[10px] leading-none text-muted-foreground font-mono">
              <span>
                {progress.current_frame} / {progress.total_frames} segments
              </span>
              <span>
                {progress.processing_rate && progress.processing_rate > 0 && progress.total_frames
                  ? `ETA: ${((progress.total_frames - progress.current_frame) / progress.processing_rate / 60).toFixed(1)} min`
                  : `${progress.progress_percent.toFixed(0)}%`}
              </span>
            </div>
            <Progress value={progress.progress_percent} className="h-1.5" />
            {progress.processing_rate && progress.processing_rate > 0 && (
              <div className="text-[10px] text-center text-muted-foreground font-mono">
                {progress.processing_rate.toFixed(1)} seg/s
              </div>
            )}
          </div>
        )}

        {/* Detailed Session Metrics */}
        {isRunning && sessionMetrics && (
          <div className="grid grid-cols-2 gap-2 text-[10px] text-muted-foreground font-mono bg-muted/20 p-2 rounded border border-border/50">
            <div>Elapsed: {sessionMetrics.elapsedSeconds.toFixed(1)}s</div>
            <div>Rate: {sessionMetrics.processingRate.toFixed(1)} seg/s</div>
            <div>Queue: {sessionMetrics.queueDepth}</div>
            <div>
              Segments: {sessionMetrics.segmentsProcessed}
              {sessionMetrics.segmentsTotal
                ? ` / ${sessionMetrics.segmentsTotal}`
                : ""}
            </div>
            {sessionMetrics.batchSize &&
              sessionMetrics.batchSize > 1 &&
              sessionMetrics.segmentsTotal && (
                <div className="col-span-2 text-blue-500">
                  Batches Left:{" "}
                  {Math.ceil(
                    Math.max(
                      0,
                      sessionMetrics.segmentsTotal -
                        sessionMetrics.segmentsProcessed
                    ) / sessionMetrics.batchSize
                  )}
                </div>
              )}
          </div>
        )}
      </div>

      <ReportModal
        sessionId={analysisJobId || undefined}
        open={reportModalOpen}
        onOpenChange={setReportModalOpen}
      />
    </div>
  );
}
