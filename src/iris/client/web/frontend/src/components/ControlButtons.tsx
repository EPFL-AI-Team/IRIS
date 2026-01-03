import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Play, Square, FileText } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { ReportModal } from "@/components/ReportModal";

export function ControlButtons() {
  const [reportModalOpen, setReportModalOpen] = useState(false);

  // Get state from store
  const isStreaming = useAppStore((state) => state.isStreaming);
  const setIsStreaming = useAppStore((state) => state.setIsStreaming);
  const addLog = useAppStore((state) => state.addLog);
  const segmentConfig = useAppStore((state) => state.segmentConfig);
  const cameraMode = useAppStore((state) => state.cameraMode);
  const requestClientCamera = useAppStore((state) => state.requestClientCamera);
  const setSessionState = useAppStore((state) => state.setSessionState);
  const sessionState = useAppStore((state) => state.sessionState);
  const analysisJobId = useAppStore((state) => state.analysisJobId);

  const handleStart = async () => {
    try {
      // Common Step: Initialize Session
      addLog("Initializing session...", "INFO");
      const initResponse = await fetch("/api/session/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frames_per_segment: segmentConfig.framesPerSegment,
          overlap_frames: segmentConfig.overlapFrames,
          camera_mode: cameraMode,
        }),
      });

      if (!initResponse.ok) {
        throw new Error(`Init failed: ${initResponse.statusText}`);
      }

      const initData = await initResponse.json();
      const sessionId = initData.session_id;
      setSessionState({ sessionId, configured: true });
      addLog(`Session initialized: ${sessionId}`, "INFO");

      // A. Handle Browser Camera Mode (Client-side)
      if (cameraMode === "client") {
        setIsStreaming(true);
        requestClientCamera(); // Ensure camera is active
        toast.success("Browser streaming started");
        return;
      }

      // B. Handle Server Camera Mode (Backend-side)
      // Note: We might want to pass session_id to /api/start in the future,
      // but for now keeping it compatible with existing endpoint
      addLog("Starting server streaming...", "INFO");

      // 2. Send config in the POST body
      const response = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frames_per_segment: segmentConfig.framesPerSegment,
          overlap_frames: segmentConfig.overlapFrames,
        }),
      });

      const data = await response.json();

      if (data.status === "ok") {
        setIsStreaming(true); // Update UI state
        addLog("Streaming started successfully", "INFO");
        toast.success("Streaming started");
      } else {
        addLog(`Start failed: ${data.message}`, "ERROR");
        toast.error(`Failed to start: ${data.message}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      addLog(`Failed to start streaming: ${message}`, "ERROR");
      toast.error("Failed to start streaming");
    }
  };

  const handleStop = async () => {
    // Stop both modes
    setIsStreaming(false);

    if (cameraMode === "server") {
      try {
        addLog("Stopping streaming...", "INFO");
        const response = await fetch("/api/stop", { method: "POST" });
        const data = await response.json();

        if (data.status === "ok") {
          addLog("Streaming stopped successfully", "INFO");
          toast.info("Streaming stopped");
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown error";
        addLog(`Failed to stop streaming: ${message}`, "ERROR");
        toast.error("Failed to stop streaming");
      }
    } else {
      toast.info("Browser streaming stopped");
    }
  };

  // Determine if we have a session to generate report for
  const hasSession = sessionState.sessionId || analysisJobId;

  return (
    <>
      <div className="flex gap-2">
        <Button onClick={handleStart} disabled={isStreaming} className="flex-1">
          <Play className="w-4 h-4 mr-2" />
          Start
        </Button>
        <Button
          variant="destructive"
          onClick={handleStop}
          disabled={!isStreaming}
          className="flex-1"
        >
          <Square className="w-4 h-4 mr-2" />
          Stop
        </Button>
        <Button
          variant="outline"
          onClick={() => setReportModalOpen(true)}
          disabled={!hasSession}
          title={hasSession ? "Generate report for current session" : "No session available"}
        >
          <FileText className="w-4 h-4" />
        </Button>
      </div>
      <ReportModal
        sessionId={sessionState.sessionId || analysisJobId || undefined}
        open={reportModalOpen}
        onOpenChange={setReportModalOpen}
      />
    </>
  );
}
