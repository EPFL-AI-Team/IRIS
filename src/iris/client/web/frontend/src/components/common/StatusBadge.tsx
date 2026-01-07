import { Badge } from "@/components/ui/badge";
import type { ConnectionStatus } from "../../types/";

/**
 * Status badge with color-coded variants.
 */
interface StatusBadgeProps {
  status: ConnectionStatus;
  label?: string; // Optional label for compact display
  onClick?: () => void;
}

export function StatusBadge({ status, label, onClick }: StatusBadgeProps) {
  const displayText =
    status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();

  const getVariant = ():
    | "default"
    | "secondary"
    | "destructive"
    | "outline" => {
    switch (status.toLowerCase()) {
      case "connected":
        return "default";
      case "connecting":
        return "secondary";
      case "disconnected":
      case "error":
        return "destructive";
      default:
        return "outline";
    }
  };

  const isClickable =
    typeof onClick === "function" &&
    (status.toLowerCase() === "disconnected" ||
      status.toLowerCase() === "error");

  return (
    <Badge
      variant={getVariant()}
      asChild={isClickable}
      className={isClickable ? "cursor-pointer select-none" : undefined}
    >
      {isClickable ? (
        <button type="button" onClick={onClick}>
          {label ? `${label}: ${displayText}` : displayText}
        </button>
      ) : (
        <span>{label ? `${label}: ${displayText}` : displayText}</span>
      )}
    </Badge>
  );
}
