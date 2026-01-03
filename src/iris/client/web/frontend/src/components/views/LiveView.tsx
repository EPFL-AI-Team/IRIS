import { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardAction,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useAppStore } from "../../store/useAppStore";
import { VideoCanvas } from "../VideoCanvas";
import { ResultsViewer } from "../ResultsViewer";
import { LogViewer } from "../LogViewer";
import { Trash2, FileText, Activity } from "lucide-react";

export function LiveView() {
  const [activePanel, setActivePanel] = useState<"results" | "logs">("results");
  const clearResults = useAppStore((state) => state.clearResults);
  const clearLogs = useAppStore((state) => state.clearLogs);

  const handleClearResults = async () => {
    try {
      await fetch("/api/results/clear", { method: "POST" });
      clearResults();
    } catch (error) {
      console.error("Failed to clear results:", error);
    }
  };

  return (
    // 1. CONTAINER: 'h-auto overflow-y-auto' for mobile scroll. 'lg:h-full lg:overflow-hidden' for desktop app-feel.
    <div className="grid grid-cols-12 gap-4 h-auto lg:h-full lg:min-h-0 overflow-y-auto lg:overflow-hidden p-1">
      {/* Left: Video Feed */}
      {/* 2. VIDEO COL: 'col-span-12' on mobile. 'h-[500px]' fixed height on mobile so it's visible. */}
      <div className="col-span-12 lg:col-span-8 h-125 lg:h-full flex flex-col min-h-0">
        <Card className="h-full flex flex-col border-none shadow-md overflow-hidden gap-0">
          <CardHeader className="h-10 flex flex-row items-center justify-between px-4 py-0 border-b shrink-0 bg-card space-y-0">
            <CardTitle className="text-lg font-semibold">Live Feed</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 p-0 relative flex flex-col">
            <VideoCanvas />
          </CardContent>
        </Card>
      </div>

      {/* Right: Results/Logs Panel */}
      {/* 3. PANEL COL: 'col-span-12' on mobile. 'h-auto' to allow stacking. */}
      <div className="col-span-12 lg:col-span-4 h-auto lg:h-full flex flex-col min-h-0">
        {/* Tab Switcher */}
        <div className="flex gap-1 bg-muted p-0 rounded-lg mb-2 shrink-0">
          <TabButton
            active={activePanel === "results"}
            onClick={() => setActivePanel("results")}
          >
            Results
          </TabButton>
          <TabButton
            active={activePanel === "logs"}
            onClick={() => setActivePanel("logs")}
          >
            Logs
          </TabButton>
        </div>

        {/* Panel Container */}
        {/* 4. CARD HEIGHT: 'h-[600px]' on mobile so it has space to scroll. 'lg:flex-1' on desktop to fill remaining space. */}
        <Card className="h-150 lg:h-auto lg:flex-1 flex flex-col min-h-0 shadow-md border-none overflow-hidden gap-0">
          <CardHeader className="h-10 flex flex-row items-center justify-between px-4 py-0 border-b shrink-0 bg-muted/20 space-y-0">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              {activePanel === "results" ? (
                <>
                  <Activity className="w-4 h-4 text-emerald-600" />
                  Inference Results
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4 text-blue-600" />
                  System Logs
                </>
              )}
            </CardTitle>

            <CardAction>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-destructive transition-colors"
                onClick={
                  activePanel === "results" ? handleClearResults : clearLogs
                }
                title={
                  activePanel === "results" ? "Clear Results" : "Clear Logs"
                }
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </CardAction>
          </CardHeader>

          {/* Content Area */}
          <CardContent className="flex-1 overflow-hidden p-0 bg-background">
            {activePanel === "results" ? <ResultsViewer /> : <LogViewer />}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-4 py-2 text-sm font-semibold uppercase tracking-wide rounded-md transition-all ${
        active
          ? "bg-background text-primary shadow-sm ring-1 ring-black/5"
          : "text-muted-foreground hover:text-foreground hover:bg-black/5"
      }`}
    >
      {children}
    </button>
  );
}
