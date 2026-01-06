import { useState } from "react";
import { useAppStore } from "../../store/useAppStore";
import { useAnalysisWebSocket } from "../../hooks/useAnalysisWebSocket";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
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
  // const analysisJobId = useAppStore((state) => state.analysisJobId);
  const analysisSessionMetrics = useAppStore(
    (state) => state.analysisSessionMetrics
  );
  const activeInferenceMode = useAppStore((state) => state.activeInferenceMode);
  const setActiveInferenceMode = useAppStore(
    (state) => state.setActiveInferenceMode
  );
  const { connect, disconnect } = useAnalysisWebSocket();

  const handleStart = async () => {
    if (!selectedVideo) {
      toast.error("Please select a video file");
      return;
    }

    setActiveInferenceMode("analysis"); // Lock

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
        setActiveInferenceMode(null); // Unlock on error
      }
    } catch (error) {
      console.error("Failed to start analysis:", error);
      toast.error("Failed to start analysis");
      setActiveInferenceMode(null); // Unlock on error
    }
  };

  const handleStop = async () => {
    try {
      disconnect();
      await fetch("/api/analysis/stop", { method: "POST" });
      toast.info("Analysis stopped");
      setActiveInferenceMode(null); // Unlock
    } catch (error) {
      console.error("Failed to stop analysis:", error);
      setActiveInferenceMode(null); // Unlock even on error
    }
  };

  const isRunning = analysisMode === "running";
  const isComplete = analysisMode === "complete";

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Row 1: Action Buttons */}
      <div className="flex items-center gap-2">
        {!isRunning ? (
          <>
            <Button
              onClick={handleStart}
              disabled={
                !selectedVideo ||
                (!!activeInferenceMode && activeInferenceMode !== "analysis")
              }
              size="sm"
              className="flex-1 h-8 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              <Play className="w-3.5 h-3.5 mr-1.5 fill-current" />
              {isComplete ? "Run Again" : "Start Analysis"}
            </Button>
            {activeInferenceMode === "live" && (
              <span className="text-xs text-muted-foreground whitespace-nowrap">
                Live running
              </span>
            )}
          </>
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

      {/* Row 2: Progress & Metrics */}
      <div className="flex flex-col gap-2">
        {progress && isRunning && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-[10px] leading-none text-muted-foreground font-mono">
              <span>
                {progress.current_frame} / {progress.total_frames} segments
              </span>
              <span>
                {progress.processing_rate &&
                progress.processing_rate > 0 &&
                progress.total_frames
                  ? `ETA: ${(
                      (progress.total_frames - progress.current_frame) /
                      progress.processing_rate /
                      60
                    ).toFixed(1)} min`
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
        {isRunning &&
          analysisSessionMetrics &&
          (() => {
            const batchSize = analysisSessionMetrics.batchSize || 1;
            const segmentsProcessed = analysisSessionMetrics.segmentsProcessed;
            const segmentsTotal = analysisSessionMetrics.segmentsTotal || 0;
            const batchesCompleted = Math.floor(segmentsProcessed / batchSize);
            const totalBatches = Math.ceil(segmentsTotal / batchSize);
            const segmentsInCurrentBatch = segmentsProcessed % batchSize;
            const showBatches =
              batchSize > 1 && segmentsTotal > 0 && batchesCompleted >= 1;

            return (
              <div className="grid grid-cols-2 gap-2 text-[10px] text-muted-foreground font-mono bg-muted/20 p-2 rounded border border-border/50">
                <div>
                  Elapsed: {analysisSessionMetrics.elapsedSeconds.toFixed(1)}s
                </div>
                <div className="flex items-center gap-1">
                  <span>
                    Segments: {segmentsProcessed}/{segmentsTotal}
                  </span>
                  {segmentsInCurrentBatch > 0 && batchSize > 1 && (
                    <span className="text-blue-400">
                      ({segmentsInCurrentBatch}/{batchSize} in batch)
                    </span>
                  )}
                </div>
                <div>
                  Segments: {segmentsProcessed}
                  {segmentsTotal ? ` / ${segmentsTotal}` : ""}
                </div>
                {showBatches && (
                  <div className="col-span-2 text-blue-400 flex items-center gap-1">
                    <span>
                      Batches: {batchesCompleted} / {totalBatches}
                    </span>
                    {segmentsInCurrentBatch > 0 && (
                      <span className="text-muted-foreground">
                        (processing {segmentsInCurrentBatch}/{batchSize})
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })()}
      </div>

      <ReportModal open={reportModalOpen} onOpenChange={setReportModalOpen} />
    </div>
  );
}
