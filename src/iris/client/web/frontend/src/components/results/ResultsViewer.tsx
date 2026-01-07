import { useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Camera, CheckCircle2, Terminal } from "lucide-react";
import { useAppStore } from "../../store/useAppStore";
import type { ResultItem } from "../../types";
import { cn } from "@/lib/utils";
import { formatResultAsNaturalLanguage } from "@/utils/formatResult";

// Helper to format video time
const formatVideoTime = (ms: number) => {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const millis = Math.floor(ms % 1000);
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
};

/**
 * ResultsViewer component for displaying inference results.
 */
export function ResultsViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const results = useAppStore((state) => state.results);

  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [results]);

  return (
    <div className="flex flex-col bg-muted/10">
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0 scroll-smooth"
      >
        {results.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
            <Camera className="w-12 h-12 mb-2 stroke-1" />
            <p className="text-sm">Waiting for live inference...</p>
          </div>
        ) : (
          results.map((result, index) => (
            <ResultCard key={result.id} result={result} number={index + 1} />
          ))
        )}
      </div>
    </div>
  );
}

function ResultCard({
  result,
  number,
}: {
  result: ResultItem;
  number: number;
}) {
  const isPending =
    result.status === "pending" || result.status === "processing";
  
  // Use video time if available, otherwise fall back to wall clock
  const timeDisplay = result.videoTimeMs !== undefined 
    ? formatVideoTime(result.videoTimeMs)
    : result.timestamp.toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

  // Helper to render the content
  const renderContent = () => {
    if (!result.result) return null;

    let content = result.result;

    // Clean up Assistant/System prefixes often found in VLM output
    if (content.includes("Assistant:")) {
      content = content.split("Assistant:").pop()?.trim() || content;
    }
    content = content.replace(/^```json|```$/g, "").trim();

    try {
      // Parse JSON and render as natural language
      const parsed = JSON.parse(content);
      const naturalLanguage = formatResultAsNaturalLanguage(parsed);

      return (
        <div className="mt-3 space-y-2">
          {/* Primary display: Natural language */}
          <div className="text-base font-medium text-foreground leading-relaxed bg-gradient-to-r from-emerald-50/50 to-transparent dark:from-emerald-950/20 p-3 rounded-md border-l-2 border-l-emerald-500">
            {naturalLanguage}
          </div>

          {/* Secondary: Collapsed JSON details (includes context field) */}
          <details className="text-xs group">
            <summary className="text-muted-foreground cursor-pointer hover:text-foreground transition-colors list-none flex items-center gap-2 py-1">
              <span className="inline-block transition-transform group-open:rotate-90">▶</span>
              <span>Show JSON details</span>
            </summary>
            <div className="grid grid-cols-1 gap-y-2 text-sm mt-2 bg-muted/40 p-3 rounded-md border border-border/50">
              {Object.entries(parsed).map(([key, value]) => (
                <div key={key} className="flex flex-col sm:flex-row sm:gap-4">
                  <span className="font-medium text-muted-foreground text-xs uppercase tracking-wider min-w-25 py-0.5">
                    {key}
                  </span>
                  <span className="font-mono text-foreground break-all">
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>
      );
    } catch {
      // Fallback for plain text
      return (
        <div className="mt-3 text-sm leading-relaxed text-foreground/90 bg-muted/30 p-3 rounded-md border border-border/50 whitespace-pre-wrap">
          {content}
        </div>
      );
    }
  };

  if (isPending) {
    // For pending cards, we can still show the submission time
    const submissionTime = result.submittedAt
      ? result.submittedAt.toLocaleTimeString([], {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      : "...";

    return (
      <Card className="border-l-2 border-l-yellow-500/50 bg-background shadow-sm">
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className="h-5 px-1.5 font-mono text-[10px]"
              >
                #{number}
              </Badge>
              <span className="text-xs text-muted-foreground font-mono">
                {submissionTime}
              </span>
            </div>
            <Loader2 className="w-3 h-3 animate-spin text-yellow-600" />
          </div>
          <div className="mt-2 text-xs text-muted-foreground flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500"></span>
            </span>
            Processing frames...
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      className={cn(
        "group relative overflow-hidden transition-all duration-200 hover:shadow-md border-l-4",
        "border-l-emerald-500" // Success indicator color
      )}
    >
      <div className="p-4">
        {/* Header Row */}
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold ring-1 ring-emerald-500/20">
              {number}
            </span>
            <div className="flex flex-col">
              <span className="text-[10px] font-medium uppercase text-muted-foreground tracking-wider">
                {result.videoTimeMs !== undefined ? "Video Time" : "Time"}
              </span>
              <span className="text-xs font-mono font-bold text-foreground">
                {timeDisplay}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* [NEW] Batch Badge */}
            {result.batchSize && result.batchSize > 1 && (
              <Badge variant="outline" className="text-[10px] h-5 px-1.5 text-blue-600 border-blue-200 bg-blue-50">
                Batch ({result.batchSize})
              </Badge>
            )}
            {result.inference_time !== undefined && (
              <div className="text-right">
                <div className="text-[10px] font-medium uppercase text-muted-foreground tracking-wider">
                  Inference
                </div>
                <div className="text-xs font-mono text-emerald-600 font-semibold">
                  {result.inference_time.toFixed(2)}s
                </div>
              </div>
            )}
            {result.latency !== undefined && (
              <div className="text-right">
                <div className="text-[10px] font-medium uppercase text-muted-foreground tracking-wider">
                  Latency
                </div>
                <div className="text-xs font-mono text-blue-600 font-semibold">
                  {(result.latency / 1000).toFixed(2)}s
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Content Body */}
        {renderContent()}

        {/* Footer/Meta */}
        <div className="mt-2 flex items-center justify-between">
          <Badge
            variant="secondary"
            className="text-[10px] h-5 px-1.5 text-muted-foreground bg-muted hover:bg-muted"
          >
            <Terminal className="w-3 h-3 mr-1" />
            {result.frames_processed || 1} Frame
            {(result.frames_processed || 1) !== 1 ? "s" : ""}
          </Badge>
          <CheckCircle2 className="w-4 h-4 text-emerald-500/50 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      </div>
    </Card>
  );
}
