import { useEffect } from "react";
import { Toaster, toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Sidebar } from "./components/Sidebar";
import { VideoCanvas } from "./components/VideoCanvas";
import { ResultsViewer } from "./components/ResultsViewer";
import { LogViewer } from "./components/LogViewer";
import { AnalysisView } from "./components/AnalysisView";
import { useResultsWebSocket } from "./hooks/useResultsWebSocket";
import { useAppStore } from "./store/useAppStore";
import "./index.css";

function App() {
  const addLog = useAppStore((state) => state.addLog);

  // Connect to results WebSocket (handles status updates and results)
  useResultsWebSocket();

  // Log initialization on mount
  useEffect(() => {
    addLog("IRIS Client initializing...", "INFO");
    toast.info("IRIS Client initialized");
  }, [addLog]);

  return (
    <div className="h-screen w-full flex flex-col lg:flex-row overflow-hidden bg-background text-foreground">
      {/* Sidebar - hidden on mobile, shown on desktop */}
      <aside className="hidden lg:flex lg:h-full">
        <Sidebar />
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Tabs defaultValue="live" className="flex-1 flex flex-col">
          <div className="border-b px-4 shrink-0">
            <TabsList className="h-12">
              <TabsTrigger value="live">Live Inference</TabsTrigger>
              <TabsTrigger value="analysis">Analysis & Benchmark</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="live" className="flex-1 p-4 overflow-auto m-0">
            {/* Desktop: side-by-side | Mobile: stacked */}
            <div className="flex flex-col lg:grid lg:grid-cols-2 gap-4 h-full">
              <div className="flex flex-col min-h-[300px] lg:min-h-0">
                <h2 className="text-lg font-semibold mb-2">Camera Preview</h2>
                <div className="flex-1 flex flex-col">
                  <VideoCanvas />
                </div>
              </div>
              <div className="flex flex-col overflow-hidden min-h-[300px] lg:min-h-0">
                <ResultsViewer />
              </div>
            </div>

            {/* Activity Log */}
            <div className="mt-4">
              <LogViewer />
            </div>

            {/* Mobile: show sidebar content below */}
            <div className="lg:hidden mt-4">
              <Sidebar />
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="flex-1 p-4 overflow-auto m-0">
            <AnalysisView />
          </TabsContent>
        </Tabs>
      </main>

      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;
