import { useState, useCallback, useRef, useEffect } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FileText, Loader2, Download, Copy, Check } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

interface ReportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ReportModal({
  open,
  onOpenChange,
}: ReportModalProps) {
  const [activeReportTab, setActiveReportTab] = useState<"live" | "analysis">(
    "live"
  );
  const [reportAlreadyGenerated, setReportAlreadyGenerated] = useState(false);
  const [copied, setCopied] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Hardcoded session IDs for persistent contexts
  const liveSessionId = "sess_live_default";
  const analysisSessionId = "sess_analysis_default";
  const effectiveSessionId =
    activeReportTab === "live" ? liveSessionId : analysisSessionId;

  // Use store state for report status
  const reportStatus = useAppStore((s) => s.reportStatus);
  const reportContent = useAppStore((s) => s.reportContent);
  const reportError = useAppStore((s) => s.reportError);
  const setReportStatus = useAppStore((s) => s.setReportStatus);
  const setReportContent = useAppStore((s) => s.setReportContent);
  const setReportError = useAppStore((s) => s.setReportError);

  const isGenerating = reportStatus === "generating";

  // Check for existing report on tab switch
  useEffect(() => {
    const checkExistingReport = async () => {
      try {
        const res = await fetch(`/api/report/${effectiveSessionId}`);
        if (!res.ok) {
          setReportAlreadyGenerated(false);
          setReportContent(null);
          setReportStatus("idle"); // Reset status if no report
          return;
        }
        const data = await res.json();
        if (data.report && data.report.content) {
          setReportContent(data.report.content);
          setReportStatus("ready");
          setReportAlreadyGenerated(true);
        } else {
          setReportAlreadyGenerated(false);
          setReportContent(null);
          setReportStatus("idle");
        }
      } catch (error) {
        console.error("Failed to check for existing report:", error);
        setReportAlreadyGenerated(false);
        setReportStatus("idle");
      }
    };

    if (open) {
      checkExistingReport();
    }
  }, [activeReportTab, open, effectiveSessionId, setReportContent, setReportStatus]);

  // Auto-scroll to bottom during streaming
  useEffect(() => {
    if (isGenerating && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [reportContent, isGenerating]);

  // Cleanup on unmount or close
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const generateReport = useCallback(
    async (forceRegenerate = false) => {
      if (!effectiveSessionId) {
        setReportError("No session ID available. Run analysis first.");
        setReportStatus("error");
        return;
      }

      setReportStatus("generating");
      setReportContent("");
      setReportError(null);
      setReportAlreadyGenerated(false); // Reset this flag when generating new

      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/api/report/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: effectiveSessionId,
            force_regenerate: forceRegenerate,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        // Check content type - handle both JSON and streaming responses
        const contentType = response.headers.get("content-type") || "";

        if (contentType.includes("application/json")) {
          // JSON response (fallback report)
          const data = await response.json();
          if (data.report) {
            setReportContent(data.report);
          } else if (data.error) {
            throw new Error(data.error);
          }
          setReportStatus("ready");
        } else {
          // Streaming response (Gemini)
          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error("Streaming not supported");
          }

          const decoder = new TextDecoder();
          let done = false;
          let accumulated = "";

          while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
              const text = decoder.decode(value, { stream: true });
              accumulated += text;
              setReportContent(accumulated);
            }
          }

          setReportStatus("ready");
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          setReportStatus("idle");
          return;
        }
        const errorMsg =
          err instanceof Error ? err.message : "Failed to generate report";
        setReportError(errorMsg);
        setReportStatus("error");
      } finally {
        abortControllerRef.current = null;
      }
    },
    [effectiveSessionId, setReportStatus, setReportContent, setReportError]
  );

  const generateFallbackReport = useCallback(async () => {
    if (!effectiveSessionId) {
      setReportError("No session ID available. Run analysis first.");
      setReportStatus("error");
      return;
    }

    setReportStatus("generating");
    setReportContent("");
    setReportError(null);

    try {
      const response = await fetch(
        `/api/report/fallback/${effectiveSessionId}`
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setReportContent(data.report);
      setReportStatus("ready");
    } catch (err) {
      const errorMsg =
        err instanceof Error ? err.message : "Failed to generate report";
      setReportError(errorMsg);
      setReportStatus("error");
    }
  }, [effectiveSessionId, setReportStatus, setReportContent, setReportError]);

  const handleCancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setReportStatus("idle");
  }, [setReportStatus]);

  const handleCopy = useCallback(async () => {
    if (reportContent) {
      await navigator.clipboard.writeText(reportContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [reportContent]);

  const handleDownload = useCallback(() => {
    if (reportContent) {
      const blob = new Blob([reportContent], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis-report-${effectiveSessionId || "unknown"}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }, [reportContent, effectiveSessionId]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Analysis Report
          </DialogTitle>
          <DialogDescription>
            Generate an AI-powered analysis report for your session.
          </DialogDescription>
        </DialogHeader>

        {/* Tab Switcher */}
        <div className="flex gap-1 bg-muted p-1 rounded-lg mb-2">
          <button
            onClick={() => setActiveReportTab("live")}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
              activeReportTab === "live"
                ? "bg-background text-foreground shadow"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Live Mode
          </button>
          <button
            onClick={() => setActiveReportTab("analysis")}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
              activeReportTab === "analysis"
                ? "bg-background text-foreground shadow"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Analysis Mode
          </button>
        </div>

        <div className="flex items-center gap-4 py-2">
          <div className="flex gap-2">
            {isGenerating ? (
              <Button variant="destructive" onClick={handleCancel}>
                Cancel
              </Button>
            ) : reportAlreadyGenerated ? (
              <div className="flex items-center gap-2">
                <Badge variant="secondary">Already Generated</Badge>
                <Button
                  variant="outline"
                  onClick={() => generateReport(true)} // Force regenerate
                >
                  Regenerate
                </Button>
              </div>
            ) : (
              <>
                <Button onClick={() => generateReport(false)}>
                  Generate Report
                </Button>
                <Button variant="outline" onClick={generateFallbackReport}>
                  Basic Stats
                </Button>
              </>
            )}
          </div>

          {reportContent && !isGenerating && (
            <div className="flex gap-2 ml-auto">
              <Button variant="ghost" size="icon" onClick={handleCopy}>
                {copied ? (
                  <Check className="h-4 w-4 text-green-500" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
              <Button variant="ghost" size="icon" onClick={handleDownload}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {reportError && (
          <div className="bg-destructive/10 text-destructive px-4 py-2 rounded-md text-sm">
            {reportError}
          </div>
        )}

        <div
          ref={contentRef}
          className="flex-1 overflow-auto bg-background rounded-md border p-4"
        >
          {isGenerating && !reportContent && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating report...
            </div>
          )}
          {reportContent ? (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-foreground prose-p:text-foreground prose-li:text-foreground prose-strong:text-foreground prose-code:text-foreground prose-pre:bg-muted prose-code:bg-muted/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded">
              <Markdown remarkPlugins={[remarkGfm]}>{reportContent}</Markdown>
              {isGenerating && (
                <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1" />
              )}
            </div>
          ) : (
            <span className="text-muted-foreground">
              Click "Generate Report" for an AI-powered analysis or "Basic
              Stats" for a quick summary.
            </span>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
