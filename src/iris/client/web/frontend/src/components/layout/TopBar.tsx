import { useState, useEffect, useRef } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Play,
  Square,
  FileText,
  RotateCcw,
  Trash2,
  Settings,
  Wifi,
  WifiOff,
  Loader2,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useClientWebSocket } from "@/hooks/useClientWebSocket";
import { SegmentSettings } from "@/components/SegmentSettings";
import { ReportModal } from "@/components/ReportModal";

/**
 * TopBar component - expanded top bar with all controls.
 *
 * Layout:
 * - Row 1: Brand + Tab Switcher + Connection Status + Session Info + Reset Button
 * - Row 2: Segment Settings (inline) + Start/Stop/Clear Queue/Report buttons
 * - Row 3: Session Metrics (when inference active)
 */
export function TopBar() {
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [showServerConfig, setShowServerConfig] = useState(false);
  const [clientElapsedSeconds, setClientElapsedSeconds] = useState(0);
  const streamingStartTimeRef = useRef<number | null>(null);

  // Store state
  const activeTab = useAppStore((state) => state.activeTab);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const connectionStatus = useAppStore((state) => state.connectionStatus);
  const serverAlive = useAppStore((state) => state.serverAlive);
  const isStreaming = useAppStore((state) => state.isStreaming);
  const setIsStreaming = useAppStore((state) => state.setIsStreaming);
  const sessionState = useAppStore((state) => state.sessionState);
  const sessionMetrics = useAppStore((state) => state.sessionMetrics);
  // const segmentConfig = useAppStore((state) => state.segmentConfig);
  const addLog = useAppStore((state) => state.addLog);
  const analysisJobId = useAppStore((state) => state.analysisJobId);
  const serverConfig = useAppStore((state) => state.serverConfig);
  const setServerConfig = useAppStore((state) => state.setServerConfig);
  const reportStatus = useAppStore((state) => state.reportStatus);

  // WebSocket connection for sending commands
  const { startInference, stopInference, clearQueue, resetSession } =
    useClientWebSocket();

  // Client-side elapsed timer
  useEffect(() => {
    if (!isStreaming) {
      return;
    }

    // Set start time when streaming begins (only once)
    if (!streamingStartTimeRef.current) {
      streamingStartTimeRef.current = Date.now();
    }

    // Update elapsed time every second
    const interval = setInterval(() => {
      if (streamingStartTimeRef.current) {
        const elapsed = Math.floor((Date.now() - streamingStartTimeRef.current) / 1000);
        setClientElapsedSeconds(elapsed);
      }
    }, 1000);

    return () => {
      clearInterval(interval);
      // Reset on cleanup when streaming stops
      streamingStartTimeRef.current = null;
      setClientElapsedSeconds(0);
    };
  }, [isStreaming]);

  // Server config form state
  const [host, setHost] = useState(serverConfig.host);
  const [port, setPort] = useState(serverConfig.port);
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [geminiKeyConfigured, setGeminiKeyConfigured] = useState(false);

  // Check if Gemini API key is configured on mount
  useEffect(() => {
    fetch("/api/config/gemini-key")
      .then(res => res.json())
      .then(data => setGeminiKeyConfigured(data.configured))
      .catch(() => setGeminiKeyConfigured(false));
  }, []);

  const handleStart = () => {
    addLog("Starting inference...", "INFO");
    startInference();
    setIsStreaming(true);
    toast.success("Inference started");
  };

  const handleStop = () => {
    addLog("Stopping inference...", "INFO");
    stopInference();
    setIsStreaming(false);
    toast.info("Inference stopped");
  };

  const handleClearQueue = () => {
    addLog("Clearing inference queue...", "INFO");
    clearQueue();
    toast.success("Queue cleared");
  };

  const handleResetSession = () => {
    addLog("Resetting session...", "INFO");
    resetSession();
    toast.success("Session reset");
  };

  const handleServerConfigSubmit = async () => {
    try {
      const response = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host, port, endpoint: serverConfig.endpoint }),
      });

      if (response.ok) {
        setServerConfig({ ...serverConfig, host, port });
        setShowServerConfig(false);
        toast.success("Server config updated");
      } else {
        toast.error("Failed to update config");
      }
    } catch {
      toast.error("Failed to update config");
    }
  };

  const handleGeminiKeySubmit = async () => {
    try {
      const response = await fetch("/api/config/gemini-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: geminiApiKey }),
      });

      if (response.ok) {
        setGeminiKeyConfigured(!!geminiApiKey);
        setGeminiApiKey(""); // Clear input after saving
        toast.success("Gemini API key updated");
      } else {
        toast.error("Failed to update API key");
      }
    } catch {
      toast.error("Failed to update API key");
    }
  };

  // Determine if we have a session for report generation
  const hasSession = sessionState.sessionId || analysisJobId;

  return (
    <div className="bg-background border-b sticky top-0 z-50">
      {/* Row 1: Brand + Tabs + Status + Session */}
      <div className="flex items-center justify-between px-4 py-2 gap-4">
        {/* Left: Brand + Tabs */}
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold tracking-tight">IRIS</h1>

          {/* Tab Switcher */}
          <div className="flex gap-1 bg-muted p-1 rounded-lg">
            <TabButton
              active={activeTab === "live"}
              onClick={() => setActiveTab("live")}
            >
              Live
            </TabButton>
            <TabButton
              active={activeTab === "analysis"}
              onClick={() => setActiveTab("analysis")}
            >
              Analysis
            </TabButton>
          </div>
        </div>

        {/* Center: Status Badges */}
        <div className="flex items-center gap-2">
          {/* Connection Status */}
          <Badge
            variant={
              connectionStatus === "connected" ? "default" : "destructive"
            }
            className="gap-1"
          >
            {connectionStatus === "connected" ? (
              <Wifi className="w-3 h-3" />
            ) : (
              <WifiOff className="w-3 h-3" />
            )}
            {connectionStatus === "connected" ? "Connected" : connectionStatus}
          </Badge>

          {/* Inference Server Status */}
          <Badge variant={serverAlive ? "default" : "secondary"}>
            Server: {serverAlive ? "Online" : "Offline"}
          </Badge>

          {/* Report Status Badge */}
          {reportStatus !== "idle" && (
            <Badge
              variant={
                reportStatus === "ready"
                  ? "default"
                  : reportStatus === "generating"
                    ? "secondary"
                    : "destructive"
              }
              className="gap-1"
            >
              {reportStatus === "generating" && (
                <Loader2 className="w-3 h-3 animate-spin" />
              )}
              {reportStatus === "ready" && <FileText className="w-3 h-3" />}
              Report: {reportStatus.charAt(0).toUpperCase() + reportStatus.slice(1)}
            </Badge>
          )}

          {/* Server Config Button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setShowServerConfig(!showServerConfig)}
          >
            <Settings className="w-4 h-4" />
          </Button>
        </div>

        {/* Right: Session Info + Reset */}
        <div className="flex items-center gap-2">
          {sessionState.sessionId && (
            <Badge variant="outline" className="font-mono text-xs">
              Session: {sessionState.sessionId.slice(0, 8)}
            </Badge>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleResetSession}
            disabled={isStreaming}
            title="Reset session"
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            Reset
          </Button>
        </div>
      </div>

      {/* Server Config Panel (collapsible) */}
      {showServerConfig && (
        <div className="px-4 py-2 bg-muted/50 border-t space-y-2">
          {/* Inference Server Config */}
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              Inference Server:
            </span>
            <input
              type="text"
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="Host"
              className="h-8 w-32 rounded-md border border-input bg-background px-2 text-sm"
            />
            <input
              type="number"
              value={port}
              onChange={(e) => setPort(parseInt(e.target.value) || 8005)}
              placeholder="Port"
              className="h-8 w-20 rounded-md border border-input bg-background px-2 text-sm"
            />
            <Button size="sm" onClick={handleServerConfigSubmit}>
              Update
            </Button>
          </div>

          {/* Gemini API Key Config */}
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">
              Gemini API Key:
            </span>
            <input
              type="password"
              value={geminiApiKey}
              onChange={(e) => setGeminiApiKey(e.target.value)}
              placeholder={geminiKeyConfigured ? "••••••••" : "Enter API key"}
              className="h-8 w-64 rounded-md border border-input bg-background px-2 text-sm"
            />
            <Button size="sm" onClick={handleGeminiKeySubmit}>
              {geminiKeyConfigured ? "Update" : "Set"}
            </Button>
            {geminiKeyConfigured && (
              <Badge variant="default" className="text-xs">
                Configured
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Row 2: Segment Settings + Controls */}
      <div className="flex items-center justify-between px-4 py-2 border-t gap-4">
        {/* Left: Segment Settings */}
        <div className="flex-1">
          <SegmentSettings disabled={isStreaming} />
        </div>

        {/* Right: Control Buttons */}
        <div className="flex items-center gap-2">
          <Button
            onClick={handleStart}
            disabled={isStreaming || connectionStatus !== "connected"}
            size="sm"
          >
            <Play className="w-4 h-4 mr-1" />
            Start
          </Button>
          <Button
            variant="destructive"
            onClick={handleStop}
            disabled={!isStreaming}
            size="sm"
          >
            <Square className="w-4 h-4 mr-1" />
            Stop
          </Button>
          <Button
            variant="outline"
            onClick={handleClearQueue}
            disabled={!isStreaming}
            size="sm"
            title="Clear inference queue"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
          <Separator orientation="vertical" className="h-6" />
          <Button
            variant="outline"
            onClick={() => setReportModalOpen(true)}
            disabled={!hasSession}
            size="sm"
            title="Generate report"
          >
            <FileText className="w-4 h-4 mr-1" />
            Report
          </Button>
        </div>
      </div>

      {/* Row 3: Session Metrics (when active) */}
      {sessionMetrics && isStreaming && (
        <div className="px-4 py-1.5 bg-muted/30 border-t flex items-center gap-6 text-sm">
          <span className="text-muted-foreground">
            Elapsed:{" "}
            <span className="font-mono text-foreground">
              {formatDuration(sessionMetrics.elapsedSeconds)}
            </span>
          </span>
          <span className="text-muted-foreground">
            Segments:{" "}
            <span className="font-mono text-foreground">
              {sessionMetrics.segmentsProcessed}
              {sessionMetrics.segmentsTotal &&
                `/${sessionMetrics.segmentsTotal}`}
            </span>
          </span>
          <span className="text-muted-foreground">
            Queue:{" "}
            <span className="font-mono text-foreground">
              {sessionMetrics.queueDepth}
            </span>
          </span>
          <span className="text-muted-foreground">
            Rate:{" "}
            <span className="font-mono text-foreground">
              {sessionMetrics.processingRate.toFixed(1)}/s
            </span>
          </span>
        </div>
      )}

      <ReportModal
        sessionId={sessionState.sessionId || analysisJobId || undefined}
        open={reportModalOpen}
        onOpenChange={setReportModalOpen}
      />
    </div>
  );
}

/**
 * Tab button component for the tab switcher.
 */
function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
        active
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground"
      }`}
    >
      {children}
    </button>
  );
}

/**
 * Format duration in seconds to MM:SS format.
 */
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
