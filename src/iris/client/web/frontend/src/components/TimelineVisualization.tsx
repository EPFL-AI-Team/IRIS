import { useMemo } from "react";
import { useAppStore } from "../store/useAppStore";

interface TimelineCard {
  id: string;
  startMs: number;
  endMs: number;
  xPercent: number;
  widthPercent: number;
  overlapDepth: number;
  color: string;
  type: "ground_truth" | "inference";
  label: string;
  sublabel: string;
  details: Record<string, string>;
}

/**
 * Compute overlap depth for cards to enable stacking visualization.
 * Cards that overlap in time are assigned increasing depth values.
 */
function computeOverlapDepths(
  cards: Array<{ startMs: number; endMs: number }>
): number[] {
  const depths: number[] = new Array(cards.length).fill(0);
  const activeIntervals: Array<{ endMs: number; depth: number }> = [];

  // Sort cards by start time
  const sortedIndices = cards
    .map((_, i) => i)
    .sort((a, b) => cards[a].startMs - cards[b].startMs);

  for (const idx of sortedIndices) {
    const card = cards[idx];

    // Remove intervals that have ended
    const stillActive = activeIntervals.filter((iv) => iv.endMs > card.startMs);

    // Find the next available depth
    const usedDepths = new Set(stillActive.map((iv) => iv.depth));
    let depth = 0;
    while (usedDepths.has(depth)) {
      depth++;
    }

    depths[idx] = depth;
    stillActive.push({ endMs: card.endMs, depth });
    activeIntervals.length = 0;
    activeIntervals.push(...stillActive);
  }

  return depths;
}

/**
 * Timeline visualization component with card-based segments.
 * Shows ground truth and inference results with overlap stacking.
 * Color coding: blue (ground truth), green (match), red (mismatch), gray (no GT).
 */
export function TimelineVisualization() {
  const groundTruth = useAppStore((state) => state.groundTruthAnnotations);
  const results = useAppStore((state) => state.analysisResults);
  const currentPosition = useAppStore((state) => state.currentPlaybackPosition);
  const datasets = useAppStore((state) => state.availableDatasets);
  const selectedVideo = useAppStore((state) => state.selectedVideoFile);
  const setCurrentPlaybackPosition = useAppStore(
    (state) => state.setCurrentPlaybackPosition
  );
  const segmentConfig = useAppStore((state) => state.segmentConfig);

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
    return Math.max((durationMs / videoDuration) * 100, 0.5); // Min 0.5% width
  };

  // Determine segment color based on match with ground truth
  const getSegmentColor = (
    result: (typeof results)[0]
  ): { color: string; status: "match" | "mismatch" | "no_gt" } => {
    const [startMs, endMs] = result.timestamp_range_ms;
    const centerMs = (startMs + endMs) / 2;

    // Find overlapping ground truth
    const gt = groundTruth.find(
      (ann) => ann.start_ms <= centerMs && centerMs <= ann.end_ms
    );

    if (!gt) return { color: "bg-gray-500", status: "no_gt" };

    // Parse inference result
    try {
      const inference = JSON.parse(result.result);

      // Compare fields
      const matches =
        inference.action === gt.action &&
        inference.tool === gt.tool &&
        inference.target === gt.target;

      return matches
        ? { color: "bg-green-500", status: "match" }
        : { color: "bg-red-500", status: "mismatch" };
    } catch {
      return { color: "bg-gray-500", status: "no_gt" };
    }
  };

  // Build inference cards with overlap detection
  const inferenceCards = useMemo((): TimelineCard[] => {
    const baseCards = results.map((result) => {
      const [startMs, endMs] = result.timestamp_range_ms;
      const { color, status } = getSegmentColor(result);

      let parsedResult: Record<string, unknown> = {};
      try {
        parsedResult = JSON.parse(result.result);
      } catch {
        parsedResult = { action: "error", tool: "?", target: "?" };
      }

      return {
        startMs,
        endMs,
        xPercent: timestampToX(startMs),
        widthPercent: durationToWidth(endMs - startMs),
        color,
        status,
        label: String(parsedResult.action || "unknown"),
        sublabel: `${parsedResult.tool || "?"} → ${parsedResult.target || "?"}`,
        details: {
          action: String(parsedResult.action || "?"),
          tool: String(parsedResult.tool || "?"),
          target: String(parsedResult.target || "?"),
          context: String(parsedResult.context || "?"),
          time: `${(startMs / 1000).toFixed(1)}s - ${(endMs / 1000).toFixed(1)}s`,
          frames: `${result.frame_range[0]}-${result.frame_range[1]}`,
          inference_time: `${(result.inference_time * 1000).toFixed(0)}ms`,
        },
      };
    });

    // Compute overlap depths
    const depths = computeOverlapDepths(
      baseCards.map((c) => ({ startMs: c.startMs, endMs: c.endMs }))
    );

    return baseCards.map((card, i) => ({
      id: `inf-${i}`,
      ...card,
      overlapDepth: depths[i],
      type: "inference" as const,
    }));
  }, [results, groundTruth, videoDuration]);

  // Build ground truth cards
  const groundTruthCards = useMemo((): TimelineCard[] => {
    return groundTruth.map((ann, i) => ({
      id: `gt-${i}`,
      startMs: ann.start_ms,
      endMs: ann.end_ms,
      xPercent: timestampToX(ann.start_ms),
      widthPercent: durationToWidth(ann.end_ms - ann.start_ms),
      overlapDepth: 0,
      color: "bg-blue-500",
      type: "ground_truth",
      label: ann.action,
      sublabel: `${ann.tool} → ${ann.target}`,
      details: {
        action: ann.action,
        tool: ann.tool,
        target: ann.target,
        context: ann.context,
        time: `${ann.start_sec.toFixed(1)}s - ${ann.end_sec.toFixed(1)}s`,
      },
    }));
  }, [groundTruth, videoDuration]);

  // Max overlap depth for height calculation
  const maxOverlapDepth = useMemo(() => {
    return Math.max(0, ...inferenceCards.map((c) => c.overlapDepth));
  }, [inferenceCards]);

  // Handle click on segment
  const handleSegmentClick = (
    event: React.MouseEvent,
    startMs: number
  ) => {
    event.stopPropagation();
    setCurrentPlaybackPosition(startMs);
  };

  // Handle click on timeline background to seek
  const handleTimelineClick = (
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    const container = event.currentTarget;
    const rect = container.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickPercent = clickX / rect.width;
    const clickTimeMs = clickPercent * videoDuration;
    setCurrentPlaybackPosition(clickTimeMs);
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

  // Calculate row heights based on overlap
  const gtRowHeight = 48;
  const infRowHeight = 48 + maxOverlapDepth * 24;

  return (
    <div className="space-y-3">
      {/* Header with segment info */}
      <div className="flex justify-between items-center text-xs text-muted-foreground">
        <span>
          Segment: {segmentConfig.segmentTime}s / {segmentConfig.framesPerSegment} frames / {segmentConfig.overlapFrames} overlap
        </span>
        <span>
          Duration: {(videoDuration / 1000).toFixed(1)}s
        </span>
      </div>

      {/* Ground Truth Row */}
      <div className="space-y-1">
        <div className="text-xs font-medium text-muted-foreground">
          Ground Truth ({groundTruthCards.length})
        </div>
        <div
          className="relative bg-muted/30 rounded-lg cursor-pointer"
          style={{ height: gtRowHeight }}
          onClick={handleTimelineClick}
        >
          {groundTruthCards.map((card) => (
            <div
              key={card.id}
              className={`absolute top-1 ${card.color} rounded-md cursor-pointer
                         hover:ring-2 hover:ring-primary transition-all group`}
              style={{
                left: `${card.xPercent}%`,
                width: `${card.widthPercent}%`,
                height: gtRowHeight - 8,
                minWidth: 4,
              }}
              onClick={(e) => handleSegmentClick(e, card.startMs)}
              title={`${card.label}\n${card.sublabel}\n${card.details.time}`}
            >
              {/* Card content - only show if wide enough */}
              {card.widthPercent > 3 && (
                <div className="px-1.5 py-1 overflow-hidden h-full flex flex-col justify-center">
                  <div className="text-[10px] font-medium text-white truncate">
                    {card.label}
                  </div>
                  {card.widthPercent > 8 && (
                    <div className="text-[9px] text-white/80 truncate">
                      {card.sublabel}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-primary pointer-events-none z-10"
            style={{ left: `${timestampToX(currentPosition)}%` }}
          />
        </div>
      </div>

      {/* Inference Results Row */}
      <div className="space-y-1">
        <div className="text-xs font-medium text-muted-foreground">
          Inference Results ({inferenceCards.length})
        </div>
        <div
          className="relative bg-muted/30 rounded-lg cursor-pointer"
          style={{ height: infRowHeight }}
          onClick={handleTimelineClick}
        >
          {inferenceCards.map((card) => {
            // Adjust opacity based on overlap depth
            const opacity = card.overlapDepth === 0 ? 1 : 0.7;
            const yOffset = 4 + card.overlapDepth * 24;

            return (
              <div
                key={card.id}
                className={`absolute ${card.color} rounded-md cursor-pointer
                           hover:ring-2 hover:ring-primary transition-all shadow-sm`}
                style={{
                  left: `${card.xPercent}%`,
                  width: `${card.widthPercent}%`,
                  top: yOffset,
                  height: 40,
                  minWidth: 4,
                  opacity,
                }}
                onClick={(e) => handleSegmentClick(e, card.startMs)}
                title={`${card.label}\n${card.sublabel}\n${card.details.time}\n${card.details.frames} frames\n${card.details.inference_time}`}
              >
                {/* Card content - only show if wide enough */}
                {card.widthPercent > 3 && (
                  <div className="px-1.5 py-1 overflow-hidden h-full flex flex-col justify-center">
                    <div className="text-[10px] font-medium text-white truncate">
                      {card.label}
                    </div>
                    {card.widthPercent > 6 && (
                      <div className="text-[9px] text-white/80 truncate">
                        {card.sublabel}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-primary pointer-events-none z-10"
            style={{ left: `${timestampToX(currentPosition)}%` }}
          />
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-500" />
          <span>Ground Truth</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span>Match</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-red-500" />
          <span>Mismatch</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-gray-500" />
          <span>No Ground Truth</span>
        </div>
      </div>
    </div>
  );
}
