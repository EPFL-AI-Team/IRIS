import { useState } from "react";
import { useAppStore } from "../store/useAppStore";
import { useAnalysisWebSocket } from "../hooks/useAnalysisWebSocket";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ReportModal } from "./ReportModal";
import { Play, Square, FileText } from "lucide-react";
import { toast } from "sonner";

/**
 * Component for controlling video analysis.
 * Includes simulation FPS input, start/stop buttons, and progress display.
 */
export function AnalysisControls() {
  const [reportModalOpen, setReportModalOpen] = useState(false);

  const analysisMode = useAppStore((state) => state.analysisMode);
  const simulationFps = useAppStore((state) => state.simulationFps);
  const setSimulationFps = useAppStore((state) => state.setSimulationFps);
  const progress = useAppStore((state) => state.analysisProgress);
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const selectedAnnotation = useAppStore(
    (state) => state.selectedAnnotationFile
  );
  const setAnalysisJobId = useAppStore((state) => state.setAnalysisJobId);
  const clearAnalysisResults = useAppStore(
    (state) => state.clearAnalysisResults
  );
  const addLog = useAppStore((state) => state.addLog);
  const analysisJobId = useAppStore((state) => state.analysisJobId);

  const { connect, disconnect } = useAnalysisWebSocket();

  const handleStart = async () => {
    if (!selectedVideo) {
      toast.error("Please select a video file");
      return;
    }

    try {
      // Call backend to start analysis
      const response = await fetch("/api/analysis/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_filename: selectedVideo,
          annotation_filename: selectedAnnotation,
          simulation_fps: simulationFps,
        }),
      });

      const data = await response.json();

      if (response.ok && data.status === "ok") {
        setAnalysisJobId(data.job_id);
        clearAnalysisResults();
        addLog(
          `Starting analysis: ${selectedVideo} (${data.total_frames} frames, ${data.annotation_count} annotations)`,
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
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium whitespace-nowrap">
            Simulation FPS:
          </label>
          <Input
            type="number"
            value={simulationFps}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setSimulationFps(Number(e.target.value))
            }
            className="w-20"
            min={1}
            max={30}
            step={0.5}
            disabled={isRunning}
          />
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

      {/* Progress Bar */}
      {progress && isRunning && (
        <div className="space-y-1">
          <Progress value={progress.progress_percent} />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Frame {progress.current_frame} / {progress.total_frames}
            </span>
            <span>{progress.progress_percent.toFixed(1)}%</span>
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
