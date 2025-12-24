import { useRef, useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import type { LogEntry } from "../types";

/**
 * LogViewer component for displaying activity logs.
 */
export function LogViewer() {
  const containerRef = useRef<HTMLDivElement>(null);

  const logs = useAppStore((state) => state.logs);
  const clearLogs = useAppStore((state) => state.clearLogs);
  const addLog = useAppStore((state) => state.addLog);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleClearLogs = () => {
    clearLogs();
    addLog("Log cleared", "INFO");
  };

  return (
    <div className="log-section">
      <div className="log-header">
        <h2>Activity Log</h2>
        <button className="btn-clear" onClick={handleClearLogs}>
          Clear Log
        </button>
      </div>
      <div id="log-container" className="log-container" ref={containerRef}>
        {logs.map((log) => (
          <LogEntryItem key={log.id} log={log} />
        ))}
      </div>
    </div>
  );
}

/**
 * Individual log entry component.
 */
function LogEntryItem({ log }: { log: LogEntry }) {
  const timestamp = log.timestamp.toLocaleTimeString();

  // Format message - handle JSON objects
  const formatMessage = (message: string) => {
    // Check if message looks like JSON
    const trimmed = message.trim();
    if (
      (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
      (trimmed.startsWith("[") && trimmed.endsWith("]"))
    ) {
      try {
        const parsed = JSON.parse(trimmed);
        return (
          <pre className="log-json">{JSON.stringify(parsed, null, 2)}</pre>
        );
      } catch {
        // Not valid JSON, return as-is
      }
    }

    // Check for markdown code blocks
    const fenceRegex = /^```(?:json)?\s*([\s\S]*?)\s*```$/i;
    const match = trimmed.match(fenceRegex);
    if (match) {
      const content = match[1].trim();
      try {
        const parsed = JSON.parse(content);
        return (
          <pre className="log-json">{JSON.stringify(parsed, null, 2)}</pre>
        );
      } catch {
        return <span>{content}</span>;
      }
    }

    return <span>{message}</span>;
  };

  return (
    <div className={`log-entry log-level-${log.level}`}>
      <span className="log-timestamp">[{timestamp}]</span>
      <span className="log-level">{log.level}</span>
      <span className="log-message">{formatMessage(log.message)}</span>
    </div>
  );
}
