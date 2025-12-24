import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAppStore } from "../../store/useAppStore";
import { VideoCanvas } from "../VideoCanvas";
import { ResultsViewer } from "../ResultsViewer";
import { LogViewer } from "../LogViewer";
import { ControlButtons } from "../ControlButtons";
import { CameraSelector } from "../CameraSelector";
import { StatusBadge } from "../StatusBadge";
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
    <div className="flex flex-col gap-4 h-full">
      {/* Top Toolbar */}
      <Card>
        <CardContent className="px-4 py-0">
          <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
            {/* Left: Camera Selector */}
            <div className="flex-1 min-w-0">
              <CameraSelector variant="horizontal" />
            </div>

            {/* Center: Control Buttons */}
            <div className="shrink-0">
              <ControlButtons />
            </div>

            {/* Right: Status Badges */}
            <div className="flex gap-2 items-center flex-wrap">
              <Badge variant="outline">FPS: {fps.toFixed(1)}</Badge>
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
          </div>
        </CardContent>
      </Card>

      {/* Main Grid: Video + Results */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0 overflow-hidden">
        {/* Video Canvas - 2/3 width */}
        <Card className="lg:col-span-2 flex flex-col overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Camera Preview</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col min-h-0">
            <VideoCanvas />
          </CardContent>
        </Card>

        {/* Results Viewer - 1/3 width */}
        <Card className="flex flex-col overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Inference Results</CardTitle>
            <CardAction>
              <Button variant="outline" size="sm" onClick={handleClearResults}>
                <Trash2 className="w-4 h-4 mr-1" />
                Clear
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden">
            <ResultsViewer />
          </CardContent>
        </Card>
      </div>

      {/* Bottom: Log Viewer */}
      <div className="shrink-0">
        <LogViewer />
      </div>
    </div>
  );
}
