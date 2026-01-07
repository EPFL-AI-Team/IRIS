import { useRef, useEffect } from "react";
import { useAppStore } from "../../store/useAppStore";
// import { ScrollArea } from "@/components/ui/scroll-area"; // Optional if you have it, else regular div works
import { cn } from "@/lib/utils";
import { Terminal } from "lucide-react";

export function LogViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const logs = useAppStore((state) => state.logs);

  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full bg-muted/10">
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto p-4 space-y-2 min-h-0 font-mono text-xs"
      >
        {logs.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
            <Terminal className="w-8 h-8 mb-2 stroke-1" />
            <p className="text-sm font-sans">No system logs yet.</p>
          </div>
        ) : (
          logs.map((log) => (
            <div
              key={log.id}
              className={cn(
                "flex gap-3 p-2 rounded border bg-background/50 hover:bg-background transition-colors",
                log.level === "ERROR" &&
                  "border-red-200 bg-red-50/50 text-red-900",
                log.level === "WARNING" &&
                  "border-yellow-200 bg-yellow-50/50 text-yellow-900",
                log.level === "INFO" && "border-border/40 text-foreground/80"
              )}
            >
              {/* Timestamp */}
              <div className="shrink-0 w-16 text-[10px] text-muted-foreground pt-0.5">
                {log.timestamp.toLocaleTimeString([], {
                  hour12: false,
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </div>

              {/* Message */}
              <div className="flex-1 wrap-break-word leading-relaxed">
                <span
                  className={cn(
                    "mr-2 font-bold text-[10px] px-1.5 py-0.5 rounded-[3px]",
                    log.level === "ERROR"
                      ? "bg-red-100 text-red-700"
                      : log.level === "WARNING"
                      ? "bg-yellow-100 text-yellow-700"
                      : "bg-blue-100 text-blue-700"
                  )}
                >
                  {log.level}
                </span>
                {log.message}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
