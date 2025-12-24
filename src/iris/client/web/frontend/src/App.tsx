import { useEffect } from "react";
import { Toaster, toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Sidebar } from "./components/Sidebar";
import { LiveView } from "./components/views/LiveView";
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
    <div className="h-screen w-full flex flex-col lg:flex-row bg-background text-foreground">
      {/* Sidebar - hidden on mobile, shown on desktop */}
      <aside className="hidden lg:flex lg:h-full overflow-y-scroll">
        <Sidebar />
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-y-scroll">
        <Tabs defaultValue="live" className="flex-1 flex flex-col">
          <div className="border-b px-4 shrink-0">
            <TabsList className="h-12 my-5">
              <TabsTrigger value="live">Live Inference</TabsTrigger>
              <TabsTrigger value="analysis">Analysis & Benchmark</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent
            value="live"
            forceMount
            className="flex-1 p-4 overflow-auto m-0"
          >
            <LiveView />

            {/* Mobile: show sidebar content below */}
            <div className="lg:hidden mt-4">
              <Sidebar />
            </div>
          </TabsContent>

          <TabsContent
            value="analysis"
            className="flex-1 p-4 overflow-auto m-0"
          >
            <AnalysisView />
          </TabsContent>
        </Tabs>
      </main>

      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;
