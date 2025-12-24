import { useState, type FormEvent } from "react";
import { useAppStore } from "../store/useAppStore";
import { useToast } from "../context/ToastContext";
import type { ConnectionStatus } from "../types";

/**
 * Sidebar component with server configuration and status display.
 */
export function Sidebar() {
  const serverConfig = useAppStore((state) => state.serverConfig);
  const sshTunnelConfig = useAppStore((state) => state.sshTunnelConfig);
  const setServerConfig = useAppStore((state) => state.setServerConfig);
  const setSSHTunnelConfig = useAppStore((state) => state.setSSHTunnelConfig);

  const [host, setHost] = useState(serverConfig.host);
  const [port, setPort] = useState(serverConfig.port);
  const [endpoint, setEndpoint] = useState(serverConfig.endpoint);
  const [tunnelHostname, setTunnelHostname] = useState(
    sshTunnelConfig.remote_host
  );

  const { showToast } = useToast();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    try {
      // Update server config
      const configResponse = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host, port, endpoint }),
      });

      // Update tunnel hostname
      const tunnelResponse = await fetch("/api/tunnel/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ remote_host: tunnelHostname }),
      });

      if (configResponse.ok && tunnelResponse.ok) {
        setServerConfig({ host, port, endpoint });
        setSSHTunnelConfig({ remote_host: tunnelHostname });
        showToast("Configuration updated successfully", "success");
      } else {
        showToast("Configuration update partially failed", "error");
      }
    } catch (error) {
      showToast("Configuration update failed", "error");
      console.error("Config update error:", error);
    }
  };

  return (
    <>
      <ConfigSection
        host={host}
        port={port}
        endpoint={endpoint}
        tunnelHostname={tunnelHostname}
        onHostChange={setHost}
        onPortChange={setPort}
        onEndpointChange={setEndpoint}
        onTunnelHostnameChange={setTunnelHostname}
        onSubmit={handleSubmit}
      />
      <StatusSection />
    </>
  );
}

/**
 * Configuration form section.
 */
function ConfigSection({
  host,
  port,
  endpoint,
  tunnelHostname,
  onHostChange,
  onPortChange,
  onEndpointChange,
  onTunnelHostnameChange,
  onSubmit,
}: {
  host: string;
  port: number;
  endpoint: string;
  tunnelHostname: string;
  onHostChange: (value: string) => void;
  onPortChange: (value: number) => void;
  onEndpointChange: (value: string) => void;
  onTunnelHostnameChange: (value: string) => void;
  onSubmit: (e: FormEvent) => void;
}) {
  return (
    <div className="config-section">
      <h2>Server Configuration</h2>
      <form id="config-form" onSubmit={onSubmit}>
        <div className="form-group">
          <label htmlFor="host">Server Host:</label>
          <input
            type="text"
            id="host"
            value={host}
            onChange={(e) => onHostChange(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="port">Server Port:</label>
          <input
            type="number"
            id="port"
            value={port}
            onChange={(e) => onPortChange(parseInt(e.target.value) || 8005)}
            min={1024}
            max={65535}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="endpoint">Endpoint:</label>
          <input
            type="text"
            id="endpoint"
            value={endpoint}
            onChange={(e) => onEndpointChange(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="tunnel-hostname">IZAR Hostname (SSH Tunnel):</label>
          <input
            type="text"
            id="tunnel-hostname"
            value={tunnelHostname}
            onChange={(e) => onTunnelHostnameChange(e.target.value)}
            placeholder="i27 or icn042.iccluster"
          />
          <small style={{ display: "block", marginTop: 4, color: "#666" }}>
            Leave empty if not using IZAR
          </small>
        </div>
        <button type="submit">Update Config</button>
      </form>
    </div>
  );
}

/**
 * Status display section.
 */
function StatusSection() {
  const isCameraActive = useAppStore((state) => state.isCameraActive);
  const isStreaming = useAppStore((state) => state.isStreaming);
  const fps = useAppStore((state) => state.fps);
  const serverConfig = useAppStore((state) => state.serverConfig);
  const previewConnection = useAppStore((state) => state.previewConnection);
  const resultsConnection = useAppStore((state) => state.resultsConnection);
  const streamingServerStatus = useAppStore(
    (state) => state.streamingServerStatus
  );

  const target = `${serverConfig.host}:${serverConfig.port}${serverConfig.endpoint}`;

  return (
    <div className="status-section">
      <h2>Status</h2>
      <div id="status">
        <p>
          <strong>Camera:</strong>{" "}
          <span id="camera-status">{isCameraActive ? "Active" : "Inactive"}</span>
        </p>
        <p>
          <strong>Streaming:</strong>{" "}
          <span id="streaming-status">
            {isStreaming ? "Active" : "Inactive"}
          </span>
        </p>
        <p>
          <strong>FPS:</strong> <span id="fps">{fps.toFixed(1)}</span>
        </p>
        <p>
          <strong>Target:</strong> <span id="target">{target}</span>
        </p>
        <p>
          <strong>Preview Connection:</strong>{" "}
          <StatusIndicator status={previewConnection} />
        </p>
        <p>
          <strong>Results Connection:</strong>{" "}
          <StatusIndicator status={resultsConnection} />
        </p>
        <p>
          <strong>Streaming Server:</strong>{" "}
          <StatusIndicator status={streamingServerStatus} />
        </p>
      </div>
    </div>
  );
}

/**
 * Status indicator component with color-coded background.
 */
function StatusIndicator({ status }: { status: ConnectionStatus }) {
  const displayText =
    status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();

  return (
    <span className={`status-indicator status-${status.toLowerCase()}`}>
      {displayText}
    </span>
  );
}
