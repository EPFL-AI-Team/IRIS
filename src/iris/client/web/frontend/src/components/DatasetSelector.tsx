import { useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * Component for selecting video and annotation files for analysis.
 * Fetches available datasets from the backend on mount.
 */
export function DatasetSelector() {
  const datasets = useAppStore((state) => state.availableDatasets);
  const setAvailableDatasets = useAppStore(
    (state) => state.setAvailableDatasets
  );
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const setSelectedVideo = useAppStore((state) => state.setSelectedVideoFile);
  const selectedAnnotation = useAppStore(
    (state) => state.selectedAnnotationFile
  );
  const setSelectedAnnotation = useAppStore(
    (state) => state.setSelectedAnnotationFile
  );
  const addLog = useAppStore((state) => state.addLog);

  useEffect(() => {
    // Fetch datasets on mount
    const fetchDatasets = async () => {
      try {
        const response = await fetch("/api/datasets");
        if (response.ok) {
          const data = await response.json();
          setAvailableDatasets(data);
          addLog(
            `Loaded ${data.videos.length} videos and ${data.annotations.length} annotations`,
            "INFO"
          );
        } else {
          addLog("Failed to fetch datasets", "ERROR");
        }
      } catch (error) {
        console.error("Failed to fetch datasets:", error);
        addLog("Failed to fetch datasets", "ERROR");
      }
    };

    fetchDatasets();
  }, [setAvailableDatasets, addLog]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="text-sm font-medium mb-2 block">Video File</label>
        <Select value={selectedVideo || ""} onValueChange={setSelectedVideo}>
          <SelectTrigger>
            <SelectValue placeholder="Select video..." />
          </SelectTrigger>
          <SelectContent>
            {datasets?.videos.map((video) => (
              <SelectItem key={video.filename} value={video.filename}>
                {video.filename} ({video.duration_sec.toFixed(1)}s,{" "}
                {video.resolution})
              </SelectItem>
            ))}
            {(!datasets || datasets.videos.length === 0) && (
              <div className="py-2 px-2 text-sm text-muted-foreground">
                No videos available
              </div>
            )}
          </SelectContent>
        </Select>
        {selectedVideo && datasets && (
          <p className="text-xs text-muted-foreground mt-1">
            {datasets.videos.find((v) => v.filename === selectedVideo)
              ?.size_mb.toFixed(1)}{" "}
            MB,{" "}
            {datasets.videos.find((v) => v.filename === selectedVideo)
              ?.frame_count}{" "}
            frames
          </p>
        )}
      </div>

      <div>
        <label className="text-sm font-medium mb-2 block">
          JSONL Annotations (Optional)
        </label>
        <Select
          value={selectedAnnotation || "__none__"}
          onValueChange={(value) =>
            setSelectedAnnotation(value === "__none__" ? null : value)
          }
        >
          <SelectTrigger>
            <SelectValue placeholder="None (optional)" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__none__">None</SelectItem>
            {datasets?.annotations.map((ann) => (
              <SelectItem key={ann.filename} value={ann.filename}>
                {ann.filename} ({ann.line_count} annotations)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selectedAnnotation && datasets && (
          <p className="text-xs text-muted-foreground mt-1">
            {datasets.annotations.find((a) => a.filename === selectedAnnotation)
              ?.size_kb.toFixed(1)}{" "}
            KB
          </p>
        )}
      </div>
    </div>
  );
}
