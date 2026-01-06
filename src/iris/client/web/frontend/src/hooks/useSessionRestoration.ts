import { useEffect } from "react";
import { useAppStore } from "../store/useAppStore";
import type { LogEntry, ResultItem } from "../types";
import type { AnalysisLog } from "../types/analysis";

/**
 * Hook to restore both live and analysis session data on app mount.
 * Fetches persisted session data from backend and populates the store.
 */
export function useSessionRestoration() {
  const setLogs = useAppStore((s) => s.setLogs);
  const setResults = useAppStore((s) => s.setResults);
  const setAnalysisLogs = useAppStore((s) => s.setAnalysisLogs);
  const setAnalysisResults = useAppStore((s) => s.setAnalysisResults);
  const setReportContent = useAppStore((s) => s.setReportContent);
  const setReportStatus = useAppStore((s) => s.setReportStatus);
  const setSegmentConfig = useAppStore((s) => s.setSegmentConfig);

  useEffect(() => {
    const restoreSessions = async () => {
      // Restore Live Session
      try {
        const liveRes = await fetch("/api/session/live/data");
        if (!liveRes.ok) {
          console.warn("Failed to fetch live session data:", liveRes.statusText);
          return;
        }

        const liveData = await liveRes.json();
        if (liveData.exists) {
          // Restore logs
          if (liveData.logs && Array.isArray(liveData.logs)) {
            const logs: LogEntry[] = liveData.logs.map((log: any) => ({
              id: `log-restored-${log.id}`,
              timestamp: new Date(log.timestamp * 1000),
              level: log.level,
              message: log.message,
            }));
            setLogs(logs);
          }

          // Restore results
          if (liveData.results && Array.isArray(liveData.results)) {
            const results: ResultItem[] = liveData.results.map((r: any) => ({
              id: `result-${r.job_id}`,
              job_id: r.job_id,
              timestamp: new Date(r.inference_start_ms),
              status: "completed" as const,
              result: r.result?.raw || JSON.stringify(r.result),
              frames_processed: r.frame_end - r.frame_start,
              inference_time: r.inference_duration_ms / 1000,
            }));
            setResults(results);
          }

          // Restore config
          if (liveData.session?.config) {
            const config = liveData.session.config;
            if (config.segment_time !== undefined && config.frames_per_segment !== undefined) {
              setSegmentConfig({
                segmentTime: config.segment_time,
                framesPerSegment: config.frames_per_segment,
                overlapFrames: config.overlap_frames || 0,
              });
            }
          }

          console.log("Live session restored:", {
            logs: liveData.logs?.length || 0,
            results: liveData.results?.length || 0,
          });
        }
      } catch (err) {
        console.error("Failed to restore live session:", err);
      }

      // Restore Analysis Session
      try {
        const analysisRes = await fetch("/api/session/analysis/data");
        if (!analysisRes.ok) {
          console.warn("Failed to fetch analysis session data:", analysisRes.statusText);
          return;
        }

        const analysisData = await analysisRes.json();
        if (analysisData.exists) {
          // Restore analysis logs
          if (analysisData.logs && Array.isArray(analysisData.logs)) {
            const analysisLogs: AnalysisLog[] = analysisData.logs.map((log: any) => ({
              id: `log-analysis-${log.id}`,
              timestamp: log.timestamp * 1000,
              video_time_ms: null,
              type: "system" as const,
              message: log.message,
            }));
            setAnalysisLogs(analysisLogs);
          }

          // Restore analysis results
          if (analysisData.results && Array.isArray(analysisData.results)) {
            const analysisResults: ResultItem[] = analysisData.results.map((r: any) => ({
              id: `result-${r.job_id}`,
              job_id: r.job_id,
              timestamp: new Date(r.inference_start_ms),
              videoTimeMs: r.video_time_ms,
              status: "completed" as const,
              result: r.result?.raw || JSON.stringify(r.result),
              frames_processed: r.frame_end - r.frame_start,
              inference_time: r.inference_duration_ms / 1000,
              frame_range: [r.frame_start, r.frame_end],
              timestamp_range_ms: [r.video_time_ms, r.video_time_ms + r.inference_duration_ms],
            }));
            setAnalysisResults(analysisResults);
          }

          // Restore report
          if (analysisData.report) {
            setReportContent(analysisData.report.content);
            setReportStatus("ready");
          }

          console.log("Analysis session restored:", {
            logs: analysisData.logs?.length || 0,
            results: analysisData.results?.length || 0,
            hasReport: !!analysisData.report,
          });
        }
      } catch (err) {
        console.error("Failed to restore analysis session:", err);
      }
    };

    restoreSessions();
  }, []); // Run once on mount
}
