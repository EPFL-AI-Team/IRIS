import { Card, CardContent, CardHeader, CardTitle, CardAction } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAppStore } from "../../store/useAppStore";
import { VideoCanvas } from "../VideoCanvas";
import { ResultsViewer } from "../ResultsViewer";
import { LogPanel } from "../LogPanel";
import { ControlButtons } from "../ControlButtons";
import { CameraSelector } from "../CameraSelector";
import { StatusBadge } from "../StatusBadge";
import { SessionMetrics } from "../SessionMetrics";
import { SegmentSettings } from "../SegmentSettings";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";

/**
 * LiveView component - Main layout for the Live Inference tab.
 * Features a dashboard-style layout with toolbar controls, video grid, and activity log.
 */
export function LiveView() {
  const fps = useAppStore((state) => state.fps);
  const previewConnection = useAppStore((state) => state.previewConnection);
  const resultsConnection = useAppStore((state) => state.resultsConnection);
  const isStreaming = useAppStore((state) => state.isStreaming);
  const requestResultsReconnect = useAppStore(
    (state) => state.requestResultsReconnect
  );
  const requestPreviewReconnect = useAppStore(
    (state) => state.requestPreviewReconnect
  );
  const addLog = useAppStore((state) => state.addLog);
  const clearResults = useAppStore((state) => state.clearResults);

  const handleClearResults = async () => {
    try {
      const response = await fetch("/api/results/clear", { method: "POST" });
      if (response.ok) {
        clearResults();
      }
    } catch (error) {
      console.error("Failed to clear results:", error);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6 h-[calc(100vh-6rem)]">
      {/* Left Sidebar - Controls & Config */}
      <div className="col-span-3 space-y-6 flex flex-col h-full overflow-hidden">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <CameraSelector variant="vertical" />
            <ControlButtons />
            <div className="flex flex-wrap gap-2">
              <StatusBadge
                status={previewConnection}
                label="Preview"
                onClick={() => {
                  requestPreviewReconnect();
                  addLog("Retrying preview connection...", "INFO");
                  toast.info("Retrying Preview connection...");
                }}
              />
              <StatusBadge
                status={resultsConnection}
                label="Results"
                onClick={() => {
                  requestResultsReconnect();
                  addLog("Retrying results connection...", "INFO");
                  toast.info("Retrying Results connection...");
                }}
              />
            </div>
          </CardContent>
        </Card>

        <div className="flex-1">
          <SegmentSettings disabled={isStreaming} />
        </div>
      </div>

      {/* Center - Video & Logs */}
      <div className="col-span-6 flex flex-col gap-6 h-full">
        <Card className="shrink-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between text-base">
              <span>Live Feed</span>
              <Badge variant="outline" className="font-mono">
                {fps.toFixed(1)} FPS
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <VideoCanvas />
            <div className="mt-4">
              <SessionMetrics />
            </div>
          </CardContent>
        </Card>

        <div className="flex-1 min-h-0">
          <LogPanel />
        </div>
      </div>

      {/* Right Sidebar - Results */}
      <div className="col-span-3 h-full flex flex-col">
        <Card className="h-full flex flex-col overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Inference Results</CardTitle>
            <CardAction>
              <Button variant="outline" size="sm" onClick={handleClearResults}>
                <Trash2 className="w-4 h-4 mr-1" />
                Clear
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden p-0">
            <ResultsViewer />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
