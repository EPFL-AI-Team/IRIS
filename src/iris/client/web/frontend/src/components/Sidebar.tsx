import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { useAppStore } from "../store/useAppStore";
import { StatusBadge } from "./StatusBadge";

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
        toast.success("Configuration updated successfully");
      } else {
        toast.error("Configuration update partially failed");
      }
    } catch (error) {
      toast.error("Configuration update failed");
      console.error("Config update error:", error);
    }
  };

  return (
    <aside className="w-full lg:w-[350px] lg:border-r bg-muted/30 p-4 flex flex-col gap-4 overflow-y-auto">
      {/* Server Configuration */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Server Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="space-y-1.5">
              <label htmlFor="host" className="text-sm font-medium">
                Server Host
              </label>
              <input
                type="text"
                id="host"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                required
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="port" className="text-sm font-medium">
                Server Port
              </label>
              <input
                type="number"
                id="port"
                value={port}
                onChange={(e) => setPort(parseInt(e.target.value) || 8005)}
                min={1024}
                max={65535}
                required
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="endpoint" className="text-sm font-medium">
                Endpoint
              </label>
              <input
                type="text"
                id="endpoint"
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
                required
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="tunnel-hostname" className="text-sm font-medium">
                IZAR Hostname (SSH Tunnel)
              </label>
              <input
                type="text"
                id="tunnel-hostname"
                value={tunnelHostname}
                onChange={(e) => setTunnelHostname(e.target.value)}
                placeholder="i27 or icn042.iccluster"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty if not using IZAR
              </p>
            </div>
            <Button type="submit" className="w-full">
              Update Config
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Status */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Status</CardTitle>
        </CardHeader>
        <CardContent>
          <StatusSection />
        </CardContent>
      </Card>
    </aside>
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
    <div className="space-y-2 text-sm">
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Camera</span>
        <Badge variant={isCameraActive ? "default" : "secondary"}>
          {isCameraActive ? "Active" : "Inactive"}
        </Badge>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Streaming</span>
        <Badge variant={isStreaming ? "default" : "secondary"}>
          {isStreaming ? "Active" : "Inactive"}
        </Badge>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">FPS</span>
        <span className="font-mono">{fps.toFixed(1)}</span>
      </div>
      <div className="flex justify-between items-center gap-2">
        <span className="text-muted-foreground shrink-0">Target</span>
        <span className="font-mono text-xs truncate">{target}</span>
      </div>
      <Separator className="my-2" />
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Preview</span>
        <StatusBadge status={previewConnection} />
      </div>
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Results</span>
        <StatusBadge status={resultsConnection} />
      </div>
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Server</span>
        <StatusBadge status={streamingServerStatus} />
      </div>
    </div>
  );
}
