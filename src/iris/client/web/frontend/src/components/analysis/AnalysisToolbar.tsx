import { Card } from "@/components/ui/card";
import { DatasetSelector } from "./DatasetSelector";
import { AnalysisControls } from "./AnalysisControls";
import { Separator } from "@/components/ui/separator";

export function AnalysisToolbar() {
  return (
    <Card className="shrink-0 border-none shadow-sm bg-card p-2">
      <div className="flex items-center gap-4 h-10 px-2">
        {/* Left: Dataset Configuration */}
        <div className="flex-1 min-w-0">
          <DatasetSelector />
        </div>

        {/* Vertical Divider */}
        <Separator orientation="vertical" className="h-6" />

        {/* Right: Analysis Controls (Start/Stop/Progress) */}
        <div className="shrink-0">
          <AnalysisControls />
        </div>
      </div>
    </Card>
  );
}
