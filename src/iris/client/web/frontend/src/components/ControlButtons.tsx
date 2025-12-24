import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Play, Square } from "lucide-react";
import { useAppStore } from "../store/useAppStore";

/**
 * ControlButtons component for Start/Stop streaming.
 */
export function ControlButtons() {
  const isStreaming = useAppStore((state) => state.isStreaming);
  const addLog = useAppStore((state) => state.addLog);

  const handleStart = async () => {
    try {
      addLog("Starting streaming...", "INFO");
      const response = await fetch("/api/start", { method: "POST" });
      const data = await response.json();

      if (data.status === "ok") {
        addLog("Streaming started successfully", "INFO");
        toast.success("Streaming started");
      } else {
        addLog(`Start failed: ${data.message}`, "ERROR");
        toast.error(`Failed to start: ${data.message}`);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      addLog(`Failed to start streaming: ${message}`, "ERROR");
      toast.error("Failed to start streaming");
    }
  };

  const handleStop = async () => {
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
  };

  return (
    <div className="flex gap-2">
      <Button
        onClick={handleStart}
        disabled={isStreaming}
        className="flex-1"
      >
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
    </div>
  );
}
