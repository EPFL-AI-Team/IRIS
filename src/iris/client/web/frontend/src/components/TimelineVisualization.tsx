import { useMemo } from "react";
import { useAppStore } from "../store/useAppStore";

/**
 * Timeline visualization component showing ground truth and inference results.
 * Two rows: ground truth (top) and inference results (bottom).
 * Color coding: blue (ground truth), green (match), red (mismatch), gray (no GT).
 */
export function TimelineVisualization() {
  const groundTruth = useAppStore((state) => state.groundTruthAnnotations);
  const results = useAppStore((state) => state.analysisResults);
  const currentPosition = useAppStore(
    (state) => state.currentPlaybackPosition
  );
  const datasets = useAppStore((state) => state.availableDatasets);
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const setCurrentPlaybackPosition = useAppStore(
    (state) => state.setCurrentPlaybackPosition
  );

  // Get video duration in milliseconds
  const videoDuration = useMemo(() => {
    const video = datasets?.videos.find((v) => v.filename === selectedVideo);
    return video ? video.duration_sec * 1000 : 60000; // Default 60s
  }, [datasets, selectedVideo]);

  // Convert timestamp (ms) to X coordinate (percentage)
  const timestampToX = (ms: number) => {
    return (ms / videoDuration) * 100;
  };

  // Convert timestamp (ms) to width (percentage)
  const durationToWidth = (durationMs: number) => {
    return (durationMs / videoDuration) * 100;
  };

  // Determine segment color based on match with ground truth
  const getSegmentColor = (result: typeof results[0]): string => {
    const [startMs, endMs] = result.timestamp_range_ms;
    const centerMs = (startMs + endMs) / 2;

    // Find overlapping ground truth
    const gt = groundTruth.find(
      (ann) => ann.start_ms <= centerMs && centerMs <= ann.end_ms
    );

    if (!gt) return "#6b7280"; // gray - no ground truth

    // Parse inference result
    try {
      const inference = JSON.parse(result.result);

      // Compare fields
      const matches =
        inference.action === gt.action &&
        inference.tool === gt.tool &&
        inference.target === gt.target;

      return matches ? "#10b981" : "#ef4444"; // green : red
    } catch (e) {
      return "#6b7280"; // gray on parse error
    }
  };

  // Handle click on timeline to seek video
  const handleTimelineClick = (
    event: React.MouseEvent<SVGSVGElement, MouseEvent>
  ) => {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickPercent = clickX / rect.width;
    const clickTimeMs = clickPercent * videoDuration;

    setCurrentPlaybackPosition(clickTimeMs);
  };

  // Handle click on segment
  const handleSegmentClick = (
    event: React.MouseEvent,
    startMs: number
  ) => {
    event.stopPropagation();
    setCurrentPlaybackPosition(startMs);
  };

  if (!selectedVideo) {
    return (
      <div className="h-32 bg-muted rounded-lg flex items-center justify-center">
        <p className="text-sm text-muted-foreground">
          Select a video to view timeline
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <svg
        width="100%"
        height="140"
        className="border rounded-lg bg-background cursor-pointer"
        onClick={handleTimelineClick}
      >
        {/* Ground Truth Row */}
        <text x="5" y="15" className="text-xs fill-foreground">
          Ground Truth
        </text>
        {groundTruth.map((ann, i) => {
          const x = timestampToX(ann.start_ms);
          const width = durationToWidth(ann.end_ms - ann.start_ms);

          return (
            <g key={`gt-${i}`}>
              <rect
                x={`${x}%`}
                width={`${width}%`}
                y="20"
                height="30"
                fill="#3b82f6"
                stroke="#1e40af"
                strokeWidth="1"
                className="cursor-pointer hover:opacity-80 transition-opacity"
                onClick={(e) => handleSegmentClick(e, ann.start_ms)}
              />
              <title>
                {`${ann.action} / ${ann.tool} / ${ann.target}\n${(ann.start_sec).toFixed(1)}s - ${(ann.end_sec).toFixed(1)}s`}
              </title>
            </g>
          );
        })}

        {/* Inference Row */}
        <text x="5" y="75" className="text-xs fill-foreground">
          Inference Results
        </text>
        {results.map((result, i) => {
          const [startMs, endMs] = result.timestamp_range_ms;
          const x = timestampToX(startMs);
          const width = durationToWidth(endMs - startMs);
          const color = getSegmentColor(result);

          let parsedResult;
          try {
            parsedResult = JSON.parse(result.result);
          } catch (e) {
            parsedResult = { action: "error", tool: "error", target: "error" };
          }

          return (
            <g key={`inf-${i}`}>
              <rect
                x={`${x}%`}
                width={`${width}%`}
                y="80"
                height="30"
                fill={color}
                stroke="#1f2937"
                strokeWidth="1"
                className="cursor-pointer hover:opacity-80 transition-opacity"
                onClick={(e) => handleSegmentClick(e, startMs)}
              />
              <title>
                {`${parsedResult.action} / ${parsedResult.tool} / ${parsedResult.target}\nFrames: ${result.frame_range[0]}-${result.frame_range[1]}\n${(startMs / 1000).toFixed(1)}s - ${(endMs / 1000).toFixed(1)}s`}
              </title>
            </g>
          );
        })}

        {/* Playback position indicator */}
        <line
          x1={`${timestampToX(currentPosition)}%`}
          x2={`${timestampToX(currentPosition)}%`}
          y1="0"
          y2="140"
          stroke="currentColor"
          strokeWidth="2"
          className="pointer-events-none stroke-primary"
        />
      </svg>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-[#3b82f6]" />
          <span>Ground Truth</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-[#10b981]" />
          <span>Match</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-[#ef4444]" />
          <span>Mismatch</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-[#6b7280]" />
          <span>No Ground Truth</span>
        </div>
      </div>
    </div>
  );
}
