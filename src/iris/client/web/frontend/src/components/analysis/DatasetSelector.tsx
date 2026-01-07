import { useEffect } from "react";
import { useAppStore } from "../../store/useAppStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Database, Loader2 } from "lucide-react";
// import { FileJson } from "lucide-react"; // Commented out - ground truth feature disabled

export function DatasetSelector() {
  const datasets = useAppStore((state) => state.availableDatasets);
  const setAvailableDatasets = useAppStore(
    (state) => state.setAvailableDatasets
  );
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const setSelectedVideo = useAppStore((state) => state.setSelectedVideoFile);
  const analysisMode = useAppStore((state) => state.analysisMode);
  const analysisSessionMetrics = useAppStore(
    (state) => state.analysisSessionMetrics
  );

  // Commented out - ground truth feature disabled
  // const selectedAnnotation = useAppStore(
  //   (state) => state.selectedAnnotationFile
  // );
  // const setSelectedAnnotation = useAppStore(
  //   (state) => state.setSelectedAnnotationFile
  // );

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch("/api/datasets");
        if (response.ok) {
          setAvailableDatasets(await response.json());
        }
      } catch (error) {
        console.error("Failed to fetch datasets:", error);
      }
    };
    fetchDatasets();
  }, [setAvailableDatasets]);

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Video Selector - Stacked */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <Database className="w-3.5 h-3.5" />
          Video Dataset
        </div>
        <Select value={selectedVideo || ""} onValueChange={setSelectedVideo}>
          <SelectTrigger className="h-8 text-xs bg-background w-full">
            <SelectValue placeholder="Select video..." />
          </SelectTrigger>
          <SelectContent>
            {datasets?.videos.map((video) => (
              <SelectItem
                key={video.filename}
                value={video.filename}
                className="text-xs"
              >
                {video.filename}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Annotation Selector - Stacked */}
      {/* COMMENTED OUT: Ground truth annotations
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <FileJson className="w-3.5 h-3.5" />
          Ground Truth (Optional)
        </div>
        <Select
          value={selectedAnnotation || "__none__"}
          onValueChange={(value) =>
            setSelectedAnnotation(value === "__none__" ? null : value)
          }
        >
          <SelectTrigger className="h-8 text-xs bg-background w-full">
            <SelectValue placeholder="None" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem
              value="__none__"
              className="text-xs text-muted-foreground"
            >
              None
            </SelectItem>
            {datasets?.annotations.map((ann) => (
              <SelectItem
                key={ann.filename}
                value={ann.filename}
                className="text-xs"
              >
                {ann.filename}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      */}

      {/* Analysis Metrics */}
      {analysisMode === "running" && analysisSessionMetrics && (
        <div className="bg-muted/30 border rounded-md p-3 space-y-2 mt-2">
          <div className="flex items-center gap-2">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span className="text-xs font-medium">Analyzing...</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="text-muted-foreground">
              Elapsed:{" "}
              <span className="text-foreground">
                {analysisSessionMetrics.elapsedSeconds}s
              </span>
            </div>
            <div className="text-muted-foreground">
              Segments:{" "}
              <span className="text-foreground">
                {analysisSessionMetrics.segmentsProcessed}/
                {analysisSessionMetrics.segmentsTotal ?? "?"}
              </span>
            </div>
            {analysisSessionMetrics.batchSize &&
              analysisSessionMetrics.batchSize > 1 && (
                <div className="col-span-2 text-blue-400">
                  Batch Mode: {analysisSessionMetrics.batchSize} segments/batch
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
}
