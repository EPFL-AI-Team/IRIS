import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileVideo, FileText } from "lucide-react";

/**
 * AnalysisView component for the Analysis & Benchmark tab.
 * Provides UI for loading video datasets and JSONL annotations,
 * video playback, and timeline visualization.
 */
export function AnalysisView() {
  return (
    <div className="space-y-4">
      {/* File Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Selection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Video File</label>
              <Button variant="outline" className="w-full justify-start">
                <FileVideo className="w-4 h-4 mr-2" />
                Select Video...
              </Button>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">JSONL Annotations</label>
              <Button variant="outline" className="w-full justify-start">
                <FileText className="w-4 h-4 mr-2" />
                Select Annotations...
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Video Player */}
      <Card>
        <CardHeader>
          <CardTitle>Video Playback</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="aspect-video bg-black rounded-lg flex items-center justify-center">
            <p className="text-muted-foreground">No video selected</p>
          </div>
        </CardContent>
      </Card>

      {/* Timeline Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-24 bg-muted rounded-lg flex items-center justify-center">
            <p className="text-muted-foreground">Timeline visualization will appear here</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
