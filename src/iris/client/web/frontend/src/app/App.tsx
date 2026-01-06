import { useEffect } from "react";
import { Toaster, toast } from "sonner";
import { TopBar } from "../components/layout/TopBar";
import { LiveView } from "../components/pages/LiveView";
import { AnalysisView } from "../components/pages/AnalysisView";
import { useClientWebSocket } from "../hooks/useClientWebSocket";
import { useSessionRestoration } from "../hooks/useSessionRestoration";
import { useAppStore } from "../store/useAppStore";

function App() {
  const addLog = useAppStore((state) => state.addLog);
  const setServerConfig = useAppStore((state) => state.setServerConfig);
  const setClientVideoConfig = useAppStore(
    (state) => state.setClientVideoConfig
  );
  const setSegmentConfig = useAppStore((state) => state.setSegmentConfig);
  const activeTab = useAppStore((state) => state.activeTab);

  // Connect to unified client WebSocket
  // Handles: session info, preview frames, results, metrics, server status
  useClientWebSocket();

  // Restore session data (live & analysis) on mount
  useSessionRestoration();

  // Load config defaults from backend on mount
  useEffect(() => {
    const loadDefaults = async () => {
      try {
        const response = await fetch("/api/config/defaults");
        if (response.ok) {
          const defaults = await response.json();

          // Set server config
          if (defaults.server) {
            setServerConfig(defaults.server);
          }

          // Set video config (backend USB camera settings)
          if (defaults.video) {
            setClientVideoConfig(defaults.video);
          }

          // Set segment config (T, s, k)
          if (defaults.segment) {
            setSegmentConfig({
              segmentTime: defaults.segment.segment_time,
              framesPerSegment: defaults.segment.frames_per_segment,
              overlapFrames: defaults.segment.overlap_frames,
            });
          }

          addLog("Loaded configuration from server", "INFO");
        }
      } catch (error) {
        console.error("Failed to load config defaults:", error);
        // Continue with hardcoded defaults
      }
    };

    loadDefaults();
    addLog("IRIS Client initializing...", "INFO");
    toast.info("IRIS Client initialized");
  }, [addLog, setServerConfig, setClientVideoConfig, setSegmentConfig]);

  return (
    <div className="h-screen w-full flex flex-col bg-background text-foreground">
      {/* TopBar - all controls in expanded top bar */}
      <TopBar />

      {/* Main Content - switches based on activeTab */}
      <main className="flex-1 overflow-auto p-4">
        {activeTab === "live" ? <LiveView /> : <AnalysisView />}
      </main>

      <Toaster position="bottom-right" richColors />
    </div>
  );
}

export default App;
