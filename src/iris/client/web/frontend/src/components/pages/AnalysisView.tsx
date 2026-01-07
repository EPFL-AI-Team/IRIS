import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DatasetSelector } from "../analysis/DatasetSelector";
import { AnalysisControls } from "../analysis/AnalysisControls";
import { VideoPlayer } from "../video/VideoPlayer";
import { LogPanel } from "../common/LogPanel";
import { TimelineVisualization } from "../analysis/TimelineVisualization";
import { Film, Activity, BarChart3, Settings2 } from "lucide-react";
import { Separator } from "@/components/ui/separator";

export function AnalysisView() {
  return (
    // 1. CONTAINER: 'h-auto overflow-y-auto' enables page scroll on mobile.
    //    'lg:h-full lg:overflow-hidden' locks it to screen height on desktop.
    <div className="flex flex-col gap-3 h-auto lg:h-full lg:min-h-0 overflow-y-auto lg:overflow-hidden p-1">
      {/* UPPER SECTION */}
      {/* 2. LAYOUT: 'h-auto' allows stacking on mobile. 'lg:flex-1' fills space on desktop. */}
      <div className="grid grid-cols-12 gap-3 h-auto lg:flex-1 lg:min-h-0">
        {/* LEFT COLUMN: Video */}
        {/* 3. VIDEO HEIGHT: Fixed 'h-[500px]' on mobile. 'lg:h-full' on desktop. */}
        <div className="col-span-12 lg:col-span-5 h-125 lg:h-full flex flex-col min-h-0">
          <Card className="h-full flex flex-col border-none shadow-md overflow-hidden gap-0">
            <CardHeader className="h-10 flex flex-row items-center justify-between px-4 py-0 border-b shrink-0 bg-card space-y-0">
              <div className="flex items-center gap-2">
                <Film className="w-4 h-4 text-blue-500" />
                <CardTitle className="text-sm font-semibold">
                  Video Playback
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="flex-1 min-h-0 p-0 relative flex flex-col">
              <VideoPlayer />
            </CardContent>
          </Card>
        </div>

        {/* RIGHT COLUMN: Controls & Logs */}
        {/* 4. SCROLLING: 'overflow-visible' on mobile (uses page scroll). 'lg:overflow-y-auto' on desktop. */}
        <div className="col-span-12 lg:col-span-7 flex flex-col h-auto lg:h-full min-h-0 gap-3 overflow-visible lg:overflow-y-auto pr-1">
          {/* Config Card */}
          <Card className="shrink-0 border-none shadow-md overflow-hidden gap-0">
            <CardHeader className="h-10 flex flex-row items-center px-4 py-0 border-b shrink-0 bg-muted/30 space-y-0">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-foreground" />
                <CardTitle className="text-sm font-semibold">
                  Configuration
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="p-3 flex flex-col gap-4">
              <DatasetSelector />
              <Separator />
              <AnalysisControls />
            </CardContent>
          </Card>

          {/* Logs Card */}
          <Card className="shrink-0 border-none shadow-md overflow-hidden gap-0">
            <CardHeader className="h-10 flex flex-row items-center justify-between px-4 py-0 border-b shrink-0 bg-muted/20 space-y-0">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-orange-500" />
                <CardTitle className="text-sm font-medium">
                  Analysis Logs
                </CardTitle>
              </div>
            </CardHeader>
            {/* Fixed height ensures it takes up space even when stacked */}
            <CardContent className="h-125 p-0 bg-background">
              <LogPanel />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* LOWER SECTION: Timeline */}
      <Card className="shrink-0 h-54 flex flex-col border-none shadow-md overflow-hidden gap-0">
        <CardHeader className="h-9 flex flex-row items-center px-4 py-0 border-b shrink-0 bg-card space-y-0">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-purple-500" />
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Timeline Visualization
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0 relative">
          <TimelineVisualization />
        </CardContent>
      </Card>
    </div>
  );
}
