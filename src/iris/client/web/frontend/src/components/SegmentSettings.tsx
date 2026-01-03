import { useAppStore } from "@/store/useAppStore";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface SegmentSettingsProps {
  disabled?: boolean;
}

/**
 * Component for configuring video analysis segment parameters.
 * T = segment time (seconds)
 * s = frames per segment
 * overlap = frames to overlap between segments
 * FPS = s / T (derived, read-only)
 */
export function SegmentSettings({ disabled = false }: SegmentSettingsProps) {
  const segmentConfig = useAppStore((state) => state.segmentConfig);
  const setSegmentConfig = useAppStore((state) => state.setSegmentConfig);
  const analysisMode = useAppStore((state) => state.analysisMode);

  const isRunning = analysisMode === "running";
  const isDisabled = isRunning || disabled;

  // Derived FPS
  const derivedFps =
    segmentConfig.segmentTime > 0
      ? segmentConfig.framesPerSegment / segmentConfig.segmentTime
      : 0;

  const handleSegmentTimeChange = (value: number) => {
    const clampedValue = Math.max(0.1, Math.min(10, value));
    setSegmentConfig({
      ...segmentConfig,
      segmentTime: clampedValue,
    });
  };

  const handleFramesPerSegmentChange = (value: number) => {
    const clampedValue = Math.max(1, Math.min(32, Math.round(value)));
    // Ensure overlap doesn't exceed frames - 1
    const maxOverlap = Math.max(0, clampedValue - 1);
    setSegmentConfig({
      ...segmentConfig,
      framesPerSegment: clampedValue,
      overlapFrames: Math.min(segmentConfig.overlapFrames, maxOverlap),
    });
  };

  const handleOverlapChange = (value: number) => {
    const maxOverlap = Math.max(0, segmentConfig.framesPerSegment - 1);
    const clampedValue = Math.max(0, Math.min(maxOverlap, Math.round(value)));
    setSegmentConfig({
      ...segmentConfig,
      overlapFrames: clampedValue,
    });
  };

  return (
    <TooltipProvider>
      <div className="flex flex-wrap gap-4 items-end">
        {/* Segment Time (T) */}
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            <Label htmlFor="segment-time" className="text-sm font-medium">
              Segment Time (T)
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3 h-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  Duration of each video segment in seconds. Each segment is
                  processed as one inference batch.
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="flex items-center gap-1">
            <Input
              id="segment-time"
              type="number"
              value={segmentConfig.segmentTime}
              onChange={(e) => handleSegmentTimeChange(Number(e.target.value))}
              className="w-20"
              min={0.1}
              max={10}
              step={0.1}
              disabled={isDisabled}
            />
            <span className="text-sm text-muted-foreground">s</span>
          </div>
        </div>

        {/* Frames per Segment (s) */}
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            <Label htmlFor="frames-per-segment" className="text-sm font-medium">
              Frames (s)
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3 h-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  Number of frames sampled per segment. These frames are sent
                  together for inference.
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
          <Input
            id="frames-per-segment"
            type="number"
            value={segmentConfig.framesPerSegment}
            onChange={(e) =>
              handleFramesPerSegmentChange(Number(e.target.value))
            }
            className="w-20"
            min={1}
            max={32}
            step={1}
            disabled={isDisabled}
          />
        </div>

        {/* Overlap Frames */}
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            <Label htmlFor="overlap-frames" className="text-sm font-medium">
              Overlap
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3 h-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  Number of frames to overlap between consecutive segments for
                  temporal continuity.
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
          <Input
            id="overlap-frames"
            type="number"
            value={segmentConfig.overlapFrames}
            onChange={(e) => handleOverlapChange(Number(e.target.value))}
            className="w-20"
            min={0}
            max={segmentConfig.framesPerSegment - 1}
            step={1}
            disabled={isDisabled}
          />
        </div>

        {/* Derived FPS (read-only) */}
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            <Label className="text-sm font-medium text-muted-foreground">
              FPS
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3 h-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  Derived frame rate: FPS = s / T. This is the effective
                  sampling rate for video processing.
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="h-9 px-3 flex items-center bg-muted rounded-md text-sm">
            {derivedFps.toFixed(1)}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
