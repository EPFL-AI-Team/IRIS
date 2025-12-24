import { useAppStore } from "../store/useAppStore";
import { useToast } from "../context/ToastContext";

/**
 * ControlButtons component for Start/Stop streaming.
 */
export function ControlButtons() {
  const isStreaming = useAppStore((state) => state.isStreaming);
  const addLog = useAppStore((state) => state.addLog);

  const { showToast } = useToast();

  const handleStart = async () => {
    try {
      addLog("Starting streaming...", "INFO");
      const response = await fetch("/api/start", { method: "POST" });
      const data = await response.json();

      if (data.status === "ok") {
        addLog("Streaming started successfully", "INFO");
        showToast("Streaming started", "success", 3000);
      } else {
        addLog(`Start failed: ${data.message}`, "ERROR");
        showToast(`Failed to start: ${data.message}`, "error");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      addLog(`Failed to start streaming: ${message}`, "ERROR");
      showToast("Failed to start streaming", "error");
    }
  };

  const handleStop = async () => {
    try {
      addLog("Stopping streaming...", "INFO");
      const response = await fetch("/api/stop", { method: "POST" });
      const data = await response.json();

      if (data.status === "ok") {
        addLog("Streaming stopped successfully", "INFO");
        showToast("Streaming stopped", "info", 3000);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      addLog(`Failed to stop streaming: ${message}`, "ERROR");
      showToast("Failed to stop streaming", "error");
    }
  };

  return (
    <div className="control-section">
      <button
        id="start-btn"
        className="btn-start"
        disabled={isStreaming}
        onClick={handleStart}
      >
        Start Streaming
      </button>
      <button
        id="stop-btn"
        className="btn-stop"
        disabled={!isStreaming}
        onClick={handleStop}
      >
        Stop Streaming
      </button>
    </div>
  );
}
