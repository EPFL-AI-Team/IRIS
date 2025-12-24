import { useRef, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, Trash2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import type { ResultItem } from "../types";

/**
 * ResultsViewer component for displaying inference results.
 * Shows pending cards for in-progress jobs and completed results.
 */
export function ResultsViewer() {
  const containerRef = useRef<HTMLDivElement>(null);

  const results = useAppStore((state) => state.results);
  const clearResults = useAppStore((state) => state.clearResults);

  // Auto-scroll to bottom when new results arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [results]);

  const handleClearResults = async () => {
    try {
      const response = await fetch("/api/results/clear", { method: "POST" });
      if (response.ok) {
        clearResults();
      }
    } catch (error) {
      console.error("Failed to clear results:", error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2 shrink-0">
        <h2 className="text-lg font-semibold">Inference Results</h2>
        <Button variant="outline" size="sm" onClick={handleClearResults}>
          <Trash2 className="w-4 h-4 mr-1" />
          Clear
        </Button>
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-3 min-h-0"
      >
        {results.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-muted-foreground text-sm">
              No results yet. Start streaming to see inference results.
            </p>
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

/**
 * Individual result card component.
 */
function ResultCard({ result, number }: { result: ResultItem; number: number }) {
  const isPending = result.status === "pending" || result.status === "processing";

  // Format timestamp
  const timestamp = result.timestamp.toLocaleTimeString();

  // Parse and format result text
  const formatResult = (resultText: string | undefined) => {
    if (!resultText) return null;

    // Remove markdown code fences
    let cleaned = resultText
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/, "")
      .replace(/```\s*$/, "")
      .trim();

    // Try to parse as JSON
    try {
      const parsed = JSON.parse(cleaned);
      return (
        <div className="font-mono text-sm space-y-1">
          {Object.entries(parsed).map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <span className="font-semibold text-muted-foreground">
                {key.charAt(0).toUpperCase() + key.slice(1)}:
              </span>
              <span>
                {typeof value === "object" && value !== null
                  ? JSON.stringify(value)
                  : String(value)}
              </span>
            </div>
          ))}
        </div>
      );
    } catch {
      // Not JSON, return as plain text
      return <p className="text-sm">{cleaned}</p>;
    }
  };

  if (isPending) {
    return (
      <Card className="border-l-4 border-l-yellow-500">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline">#{number}</Badge>
              <span className="text-xs text-muted-foreground">{timestamp}</span>
            </div>
            <Badge variant="secondary">
              {result.status === "processing" ? "Processing" : "Queued"}
            </Badge>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm italic">Processing batch...</span>
          </div>
          {result.pendingDetails && (
            <p className="text-xs text-muted-foreground mt-2">
              {result.pendingDetails}
            </p>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-l-4 border-l-green-500">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Badge variant="outline">#{number}</Badge>
            <span className="text-xs text-muted-foreground">{timestamp}</span>
          </div>
          <Badge variant="secondary">
            Frames: {result.frames_processed || 0}
          </Badge>
        </div>
        <div className="mb-2">{formatResult(result.result)}</div>
        <div className="text-xs text-muted-foreground">
          Inference: {(result.inference_time || 0).toFixed(3)}s
        </div>
      </CardContent>
    </Card>
  );
}
