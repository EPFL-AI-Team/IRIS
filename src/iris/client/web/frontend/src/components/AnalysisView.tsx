import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DatasetSelector } from "./DatasetSelector";
import { AnalysisControls } from "./AnalysisControls";
import { VideoPlayer } from "./VideoPlayer";
import { TimelineVisualization } from "./TimelineVisualization";

/**
 * AnalysisView component for the Analysis & Benchmark tab.
 * Provides UI for loading video datasets and JSONL annotations,
 * running analysis with configurable simulation FPS, video playback,
 * and timeline visualization comparing ground truth vs inference results.
 */
export function AnalysisView() {
  return (
    <div className="space-y-4">
      {/* Dataset Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Selection</CardTitle>
        </CardHeader>
        <CardContent>
          <DatasetSelector />
        </CardContent>
      </Card>

      {/* Analysis Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <AnalysisControls />
        </CardContent>
      </Card>

      {/* Video Player */}
      <Card>
        <CardHeader>
          <CardTitle>Video Playback</CardTitle>
        </CardHeader>
        <CardContent>
          <VideoPlayer />
        </CardContent>
      </Card>

      {/* Timeline Comparison */}
      <Card>
        <CardHeader>
          <CardTitle>Timeline Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <TimelineVisualization />
        </CardContent>
      </Card>
    </div>
  );
}
