import { useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
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
    <Card>
      <CardHeader className="flex flex-row items-center justify-between py-3 px-4">
        <CardTitle className="text-base">Activity Log</CardTitle>
        <Button variant="ghost" size="sm" onClick={handleClearLogs}>
          <Trash2 className="w-4 h-4 mr-1" />
          Clear
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        <div
          ref={containerRef}
          className="h-48 overflow-y-auto font-mono text-xs p-3 bg-muted/30"
        >
          {logs.map((log) => (
            <LogEntryItem key={log.id} log={log} />
          ))}
        </div>
      </CardContent>
    </Card>
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
          <pre className="whitespace-pre-wrap break-all text-muted-foreground">
            {JSON.stringify(parsed, null, 2)}
          </pre>
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
          <pre className="whitespace-pre-wrap break-all text-muted-foreground">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        );
      } catch {
        return <span>{content}</span>;
      }
    }

    return <span>{message}</span>;
  };

  const getLevelColor = () => {
    switch (log.level) {
      case "ERROR":
        return "text-red-500";
      case "WARNING":
        return "text-yellow-500";
      case "INFO":
        return "text-blue-500";
      case "DEBUG":
        return "text-gray-500";
      default:
        return "text-foreground";
    }
  };

  return (
    <div className="flex gap-2 py-0.5">
      <span className="text-muted-foreground shrink-0">[{timestamp}]</span>
      <span className={`shrink-0 font-semibold ${getLevelColor()}`}>
        {log.level}
      </span>
      <span className="break-all">{formatMessage(log.message)}</span>
    </div>
  );
}
