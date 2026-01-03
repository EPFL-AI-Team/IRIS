import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RefreshCw, Camera, Video } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import type { CameraMode } from "@/types";

/**
 * CameraSelector component for switching between browser and server cameras.
 */
interface CameraSelectorProps {
  variant?: 'vertical' | 'horizontal';
}

export function CameraSelector({ variant = 'vertical' }: CameraSelectorProps) {
  const cameraMode = useAppStore((state) => state.cameraMode);
  const setCameraMode = useAppStore((state) => state.setCameraMode);
  const clientCameraPermission = useAppStore(
    (state) => state.clientCameraPermission
  );
  const requestClientCamera = useAppStore((state) => state.requestClientCamera);
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

  const handleModeChange = (value: string) => {
    setCameraMode(value as CameraMode);
  };

  const handleEnableClientCamera = () => {
    requestClientCamera();
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

  const getPermissionStatus = () => {
    switch (clientCameraPermission) {
      case "granted":
        return "Camera access granted";
      case "denied":
        return "Camera access denied";
      default:
        return "";
    }
  };

  // Horizontal variant for toolbar
  if (variant === 'horizontal') {
    return (
      <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-center flex-wrap">
        {/* Camera Mode Select - Compact */}
        <Select value={cameraMode} onValueChange={handleModeChange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="client">
              <div className="flex items-center">
                <Camera className="w-4 h-4 mr-2" />
                Browser Camera
              </div>
            </SelectItem>
            <SelectItem value="server">
              <div className="flex items-center">
                <Video className="w-4 h-4 mr-2" />
                Server Camera
              </div>
            </SelectItem>
          </SelectContent>
        </Select>

        {/* Client Camera Options - Inline */}
        {cameraMode === "client" && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleEnableClientCamera}
          >
            <Camera className="w-4 h-4 mr-2" />
            Enable Camera
          </Button>
        )}

        {/* Server Camera Selection - Inline */}
        {cameraMode === "server" && (
          <>
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
          </>
        )}
      </div>
    );
  }

  // Vertical variant (default) for sidebar
  return (
    <div className="space-y-3">
      {/* Camera Mode Select */}
      <div className="space-y-1.5">
        <label className="text-sm font-medium">Camera Source</label>
        <Select value={cameraMode} onValueChange={handleModeChange}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="client">
              <div className="flex items-center">
                <Camera className="w-4 h-4 mr-2" />
                Browser Camera
              </div>
            </SelectItem>
            <SelectItem value="server">
              <div className="flex items-center">
                <Video className="w-4 h-4 mr-2" />
                Server Camera
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Client Camera Options */}
      {cameraMode === "client" && (
        <div className="space-y-2">
          <Button
            variant="outline"
            onClick={handleEnableClientCamera}
            className="w-full"
          >
            <Camera className="w-4 h-4 mr-2" />
            Enable Camera Access
          </Button>
          {getPermissionStatus() && (
            <p className={`text-xs ${
              clientCameraPermission === "granted"
                ? "text-green-600"
                : "text-destructive"
            }`}>
              {getPermissionStatus()}
            </p>
          )}
        </div>
      )}

      {/* Server Camera Selection */}
      {cameraMode === "server" && (
        <div className="space-y-2">
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
      )}
    </div>
  );
}
