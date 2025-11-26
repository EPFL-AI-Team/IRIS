// WebSocket for preview
let previewWs = null;
let resultsWs = null;
let statusInterval = null;

// Initialize preview WebSocket
function connectPreview() {
  previewWs = new WebSocket(`ws://${window.location.host}/preview`);

  previewWs.onmessage = (event) => {
    const img = document.getElementById("preview");
    img.src = `data:image/jpeg;base64,${event.data}`;
    document.getElementById("preview-status").textContent = "Camera active";
  };

  previewWs.onerror = () => {
    document.getElementById("preview-status").textContent = "Preview error";
  };

  previewWs.onclose = () => {
    document.getElementById("preview-status").textContent = "Camera inactive";
    setTimeout(connectPreview, 2000); // Reconnect
  };
}

// Initialize results WebSocket
function connectResults() {
  resultsWs = new WebSocket(`ws://${window.location.host}/results`);

  resultsWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    displayResult(data);
  };

  resultsWs.onerror = (error) => {
    console.error("Results WebSocket error:", error);
  };

  resultsWs.onclose = () => {
    console.log("Results WebSocket closed, reconnecting...");
    setTimeout(connectResults, 2000); // Reconnect
  };
}

// Display inference result
function displayResult(data) {
  const container = document.getElementById("results-container");

  // Format timestamp
  const timestamp = new Date(data.metrics.received_at * 1000).toLocaleTimeString();

  container.innerHTML = `
    <div class="result-content">
      <img class="result-frame" src="data:image/jpeg;base64,${data.frame}" alt="Analyzed frame">

      <div class="result-text">
        <h3>Description</h3>
        <p>${data.result}</p>
      </div>

      <div class="result-metrics">
        <h3>Metrics</h3>
        <div class="metric-row">
          <span class="metric-label">Job ID:</span>
          <span class="metric-value">${data.job_id}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Status:</span>
          <span class="metric-value">${data.status}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Inference Time:</span>
          <span class="metric-value">${data.metrics.inference_time.toFixed(3)}s</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Total Latency:</span>
          <span class="metric-value">${data.metrics.total_latency.toFixed(3)}s</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Timestamp:</span>
          <span class="metric-value">${timestamp}</span>
        </div>
      </div>
    </div>
  `;
}

// Update status display
async function updateStatus() {
  try {
    const response = await fetch("/status");
    const data = await response.json();

    document.getElementById("camera-status").textContent = data.camera_active
      ? "Active"
      : "Inactive";
    document.getElementById("streaming-status").textContent =
      data.streaming_active ? "Active" : "Inactive";
    document.getElementById("fps").textContent = data.fps.toFixed(1);
    document.getElementById(
      "target"
    ).textContent = `${data.config.server.host}:${data.config.server.port}${data.config.server.endpoint}`;

    // Update button states
    document.getElementById("start-btn").disabled = data.streaming_active;
    document.getElementById("stop-btn").disabled = !data.streaming_active;
  } catch (error) {
    console.error("Status update failed:", error);
  }
}

// Handle config form submission
document.getElementById("config-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const config = {
    host: document.getElementById("host").value,
    port: parseInt(document.getElementById("port").value),
    endpoint: document.getElementById("endpoint").value,
  };

  try {
    const response = await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });

    if (response.ok) {
      alert("Configuration updated");
      updateStatus();
    }
  } catch (error) {
    alert("Configuration update failed");
  }
});

// Start streaming
document.getElementById("start-btn").addEventListener("click", async () => {
  try {
    const response = await fetch("/start", { method: "POST" });
    const data = await response.json();

    if (data.status === "ok") {
      updateStatus();
    } else {
      alert(`Start failed: ${data.message}`);
    }
  } catch (error) {
    alert("Failed to start streaming");
  }
});

// Stop streaming
document.getElementById("stop-btn").addEventListener("click", async () => {
  try {
    const response = await fetch("/stop", { method: "POST" });
    const data = await response.json();

    if (data.status === "ok") {
      updateStatus();
    }
  } catch (error) {
    alert("Failed to stop streaming");
  }
});

// Initialize
connectPreview();
connectResults();
updateStatus();
statusInterval = setInterval(updateStatus, 1000);
