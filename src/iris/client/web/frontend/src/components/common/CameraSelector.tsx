import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RefreshCw } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

/**
 * CameraSelector component for selecting server-side cameras.
 * Note: Client-side (browser) camera support has been removed.
 */
interface CameraSelectorProps {
  variant?: 'vertical' | 'horizontal';
}

export function CameraSelector({ variant = 'vertical' }: CameraSelectorProps) {
  const serverCameras = useAppStore((state) => state.serverCameras);
  const setServerCameras = useAppStore((state) => state.setServerCameras);
  const selectedServerCamera = useAppStore(
    (state) => state.selectedServerCamera
  );
  const setSelectedServerCamera = useAppStore(
    (state) => state.setSelectedServerCamera
  );
  const addLog = useAppStore((state) => state.addLog);

  // Load server cameras on mount
  useEffect(() => {
    loadServerCameras();
  }, []);

  const loadServerCameras = async () => {
    try {
      const response = await fetch("/api/cameras");
      const data = await response.json();

      if (data.cameras && data.cameras.length > 0) {
        setServerCameras(data.cameras);
        addLog(`Found ${data.cameras.length} server cameras`, "INFO");
      } else {
        addLog("No server cameras found", "WARNING");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      addLog(`Failed to load cameras: ${message}`, "ERROR");
    }
  };

  const handleServerCameraChange = async (value: string) => {
    const cameraIndex = parseInt(value);

    try {
      const response = await fetch("/api/camera/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: cameraIndex }),
      });
      const data = await response.json();

      if (data.status === "ok") {
        setSelectedServerCamera(cameraIndex);
        addLog(`Switched to camera ${cameraIndex}`, "INFO");
      } else {
        addLog(`Failed to switch camera: ${data.message}`, "ERROR");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      addLog(`Camera switch failed: ${message}`, "ERROR");
    }
  };

  // Horizontal variant for toolbar
  if (variant === 'horizontal') {
    return (
      <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-center flex-wrap">
        <Select
          value={selectedServerCamera.toString()}
          onValueChange={handleServerCameraChange}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select camera" />
          </SelectTrigger>
          <SelectContent>
            {serverCameras.length === 0 ? (
              <SelectItem value="-1" disabled>
                No cameras found
              </SelectItem>
            ) : (
              serverCameras.map((camera) => (
                <SelectItem
                  key={camera.index}
                  value={camera.index.toString()}
                >
                  Camera {camera.index} ({camera.resolution})
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          size="icon"
          onClick={loadServerCameras}
          title="Refresh Cameras"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>
    );
  }

  // Vertical variant (default) for sidebar
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Server Camera</label>
        <Select
          value={selectedServerCamera.toString()}
          onValueChange={handleServerCameraChange}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select camera" />
          </SelectTrigger>
          <SelectContent>
            {serverCameras.length === 0 ? (
              <SelectItem value="-1" disabled>
                No cameras found
              </SelectItem>
            ) : (
              serverCameras.map((camera) => (
                <SelectItem
                  key={camera.index}
                  value={camera.index.toString()}
                >
                  Camera {camera.index} ({camera.resolution})
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>
      <Button
        variant="outline"
        onClick={loadServerCameras}
        className="w-full"
      >
        <RefreshCw className="w-4 h-4 mr-2" />
        Refresh Cameras
      </Button>
    </div>
  );
}
