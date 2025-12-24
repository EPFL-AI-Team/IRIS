import { useEffect } from "react";
import { ToastProvider } from "./context/ToastContext";
import { VideoCanvas } from "./components/VideoCanvas";
import { ResultsViewer } from "./components/ResultsViewer";
import { Sidebar } from "./components/Sidebar";
import { ControlButtons } from "./components/ControlButtons";
import { LogViewer } from "./components/LogViewer";
import { CameraSelector } from "./components/CameraSelector";
import { useResultsWebSocket } from "./hooks/useResultsWebSocket";
import { useAppStore } from "./store/useAppStore";
import "./index.css";

function AppContent() {
  const addLog = useAppStore((state) => state.addLog);

  // Connect to results WebSocket (handles status updates and results)
  useResultsWebSocket();

  // Log initialization on mount
  useEffect(() => {
    addLog("IRIS Client initializing...", "INFO");
  }, [addLog]);

  return (
    <div className="container">
      <h1>IRIS Streaming Client</h1>

      {/* Main Content: Video + Results Side by Side */}
      <div className="main-content">
        <div className="video-section">
          <h2>Camera Preview</h2>
          <VideoCanvas />
        </div>

        <div className="results-section">
          <ResultsViewer />
        </div>
      </div>

      {/* Configuration and Status Side by Side */}
      <div className="config-status-row">
        <Sidebar />
      </div>

      {/* Control Buttons */}
      <ControlButtons />

      {/* Activity Log */}
      <LogViewer />

      {/* Camera Configuration */}
      <CameraSelector />
    </div>
  );
}

function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  );
}

export default App;
