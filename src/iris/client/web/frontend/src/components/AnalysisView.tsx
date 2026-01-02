import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DatasetSelector } from "./DatasetSelector";
import { SegmentSettings } from "./SegmentSettings";
import { AnalysisControls } from "./AnalysisControls";
import { VideoPlayer } from "./VideoPlayer";
import { LogPanel } from "./LogPanel";
import { TimelineVisualization } from "./TimelineVisualization";
import { SessionMetrics } from "./SessionMetrics";

/**
 * AnalysisView component for the Analysis & Benchmark tab.
 * Provides UI for:
 * - Loading video datasets and JSONL annotations
 * - Configuring segment parameters (T, s, overlap)
 * - Running analysis with progress tracking
 * - Video playback synchronized with log panel
 * - Timeline visualization comparing ground truth vs inference results
 */
export function AnalysisView() {
  return (
    <div className="space-y-4">
      {/* Dataset Selection */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-base">Dataset Selection</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <DatasetSelector />
        </CardContent>
      </Card>

      {/* Settings and Controls Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Segment Settings */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">Segment Settings</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <SegmentSettings />
          </CardContent>
        </Card>

        {/* Analysis Controls */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">Analysis Controls</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <AnalysisControls />
          </CardContent>
        </Card>
      </div>

      {/* Session Metrics */}
      <Card>
        <CardContent className="py-2">
          <SessionMetrics />
        </CardContent>
      </Card>

      {/* Video Player and Log Panel Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Video Player */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">Video Playback</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <VideoPlayer />
          </CardContent>
        </Card>

        {/* Log Panel */}
        <div className="h-100">
          <LogPanel />
        </div>
      </div>

      {/* Timeline Comparison */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-base">Timeline Comparison</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <TimelineVisualization />
        </CardContent>
      </Card>
    </div>
  );
}
