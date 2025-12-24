import { useRef, useEffect } from "react";
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
    <>
      <div className="log-header">
        <h2>Inference Results</h2>
        <button className="btn-clear" onClick={handleClearResults}>
          Clear Results
        </button>
      </div>
      <div id="results-container" ref={containerRef}>
        {results.length === 0 ? (
          <div className="results-placeholder">
            <p>No results yet. Start streaming to see inference results.</p>
          </div>
        ) : (
          results.map((result, index) => (
            <ResultCard key={result.id} result={result} number={index + 1} />
          ))
        )}
      </div>
    </>
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
        <div
          className="formatted-result"
          style={{
            fontFamily: "monospace",
            lineHeight: 1.5,
            textAlign: "left",
            paddingLeft: 20,
          }}
        >
          {Object.entries(parsed).map(([key, value]) => (
            <div key={key}>
              <strong>{key.charAt(0).toUpperCase() + key.slice(1)}:</strong>{" "}
              {typeof value === "object" && value !== null
                ? JSON.stringify(value)
                : String(value)}
            </div>
          ))}
        </div>
      );
    } catch {
      // Not JSON, return as plain text
      return <p>{cleaned}</p>;
    }
  };

  if (isPending) {
    return (
      <div className={`result-item pending`} id={`result-${result.job_id}`}>
        <div className="result-header">
          <span className="result-number">#{number}</span>
          <span className="result-timestamp">{timestamp}</span>
          <span className="result-status">
            {result.status === "processing" ? "Processing" : "Queued"}
          </span>
        </div>
        <div className="result-text">
          <div className="loading-spinner"></div>
          <p>
            <em>Processing batch...</em>
          </p>
          {result.pendingDetails && (
            <div className="pending-details">{result.pendingDetails}</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="result-item" id={`result-${result.job_id}`}>
      <div className="result-header">
        <span className="result-number">#{number}</span>
        <span className="result-timestamp">{timestamp}</span>
        <span className="result-frames">
          Frames: {result.frames_processed || 0}
        </span>
      </div>
      <div className="result-text">{formatResult(result.result)}</div>
      <div className="result-metrics">
        <span>Inference: {(result.inference_time || 0).toFixed(3)}s</span>
      </div>
    </div>
  );
}
