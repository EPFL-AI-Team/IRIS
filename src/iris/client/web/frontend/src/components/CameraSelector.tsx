import { useEffect, type ChangeEvent } from "react";
import { useAppStore } from "../store/useAppStore";
import type { CameraMode } from "../types";

/**
 * CameraSelector component for switching between browser and server cameras.
 */
export function CameraSelector() {
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

  const handleModeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const mode = e.target.value as CameraMode;
    setCameraMode(mode);
  };

  const handleEnableClientCamera = () => {
    requestClientCamera();
  };

  const handleServerCameraChange = async (e: ChangeEvent<HTMLSelectElement>) => {
    const cameraIndex = parseInt(e.target.value);

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

  return (
    <div className="camera-config-section">
      <h2>Camera Selection</h2>

      <div className="form-group">
        <label htmlFor="camera-mode">Camera Source:</label>
        <select
          id="camera-mode"
          value={cameraMode}
          onChange={handleModeChange}
        >
          <option value="client">Browser Camera (This Device)</option>
          <option value="server">Server Camera</option>
        </select>
      </div>

      {/* Client Camera Options */}
      <div
        id="client-camera-options"
        style={{ display: cameraMode === "client" ? "block" : "none" }}
      >
        <button
          type="button"
          id="enable-client-camera-btn"
          onClick={handleEnableClientCamera}
        >
          Enable Camera Access
        </button>
        <div id="camera-permission-status">{getPermissionStatus()}</div>
      </div>

      {/* Server Camera Selection */}
      <div
        id="server-camera-options"
        style={{ display: cameraMode === "server" ? "block" : "none" }}
      >
        <div className="form-group">
          <label htmlFor="server-camera-select">Server Camera:</label>
          <select
            id="server-camera-select"
            value={selectedServerCamera}
            onChange={handleServerCameraChange}
          >
            {serverCameras.length === 0 ? (
              <option>No cameras found</option>
            ) : (
              serverCameras.map((camera) => (
                <option key={camera.index} value={camera.index}>
                  Camera {camera.index} ({camera.resolution})
                </option>
              ))
            )}
          </select>
        </div>
        <button type="button" id="refresh-cameras-btn" onClick={loadServerCameras}>
          Refresh Cameras
        </button>
      </div>
    </div>
  );
}
