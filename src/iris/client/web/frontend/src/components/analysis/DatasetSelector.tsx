import { useEffect } from "react";
import { useAppStore } from "../../store/useAppStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Database } from "lucide-react";
// import { FileJson } from "lucide-react"; // Commented out - ground truth feature disabled

export function DatasetSelector() {
  const datasets = useAppStore((state) => state.availableDatasets);
  const setAvailableDatasets = useAppStore(
    (state) => state.setAvailableDatasets
  );
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const setSelectedVideo = useAppStore((state) => state.setSelectedVideoFile);
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
    </div>
  );
}
