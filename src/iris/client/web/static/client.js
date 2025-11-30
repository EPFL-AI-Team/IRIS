// WebSocket for preview
let previewWs = null;
let resultsWs = null;
let statusInterval = null;

// Activity logging
const LOG_MAX_ENTRIES = 100;

function addLog(message, level = 'INFO') {
  const container = document.getElementById('log-container');
  if (!container) return;

  const timestamp = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = `log-entry log-level-${level}`;
  entry.innerHTML = `
    <span class="log-timestamp">[${timestamp}]</span>
    <span class="log-level">${level}</span>
    <span class="log-message">${message}</span>
  `;

  container.appendChild(entry);

  // Limit entries
  const entries = container.getElementsByClassName('log-entry');
  if (entries.length > LOG_MAX_ENTRIES) {
    container.removeChild(entries[0]);
  }

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// Update connection status
function updateConnectionStatus(elementId, status) {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = status;
    element.className = `status-indicator status-${status.toLowerCase()}`;
  }
}

// Initialize preview WebSocket
function connectPreview() {
  updateConnectionStatus('preview-connection', 'Connecting');
  addLog('Connecting to preview WebSocket...', 'INFO');
  previewWs = new WebSocket(`ws://${window.location.host}/preview`);

  previewWs.onopen = () => {
    updateConnectionStatus('preview-connection', 'Connected');
    addLog('Preview WebSocket connected', 'INFO');
  };

  previewWs.onmessage = (event) => {
    const img = document.getElementById("preview");
    img.src = `data:image/jpeg;base64,${event.data}`;
    document.getElementById("preview-status").textContent = "Camera active";
  };

  previewWs.onerror = () => {
    updateConnectionStatus('preview-connection', 'Error');
    document.getElementById("preview-status").textContent = "Preview error";
    addLog('Preview WebSocket error', 'ERROR');
  };

  previewWs.onclose = () => {
    updateConnectionStatus('preview-connection', 'Disconnected');
    document.getElementById("preview-status").textContent = "Camera inactive";
    addLog('Preview WebSocket closed, reconnecting...', 'WARNING');
    setTimeout(connectPreview, 2000); // Reconnect
  };
}

// Initialize results WebSocket
function connectResults() {
  updateConnectionStatus('results-connection', 'Connecting');
  addLog('Connecting to results WebSocket...', 'INFO');
  resultsWs = new WebSocket(`ws://${window.location.host}/results`);

  resultsWs.onopen = () => {
    updateConnectionStatus('results-connection', 'Connected');
    addLog('Results WebSocket connected', 'INFO');
  };

  resultsWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    addLog(`Received result: ${data.job_id} (${data.status})`, 'INFO');
    displayResult(data);
  };

  resultsWs.onerror = (error) => {
    updateConnectionStatus('results-connection', 'Error');
    addLog('Results WebSocket error', 'ERROR');
    console.error("Results WebSocket error:", error);
  };

  resultsWs.onclose = () => {
    updateConnectionStatus('results-connection', 'Disconnected');
    addLog('Results WebSocket closed, reconnecting...', 'WARNING');
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
    addLog('Starting streaming...', 'INFO');
    const response = await fetch("/start", { method: "POST" });
    const data = await response.json();

    if (data.status === "ok") {
      addLog('Streaming started successfully', 'INFO');
      updateStatus();
    } else {
      addLog(`Start failed: ${data.message}`, 'ERROR');
      alert(`Start failed: ${data.message}`);
    }
  } catch (error) {
    addLog(`Failed to start streaming: ${error.message}`, 'ERROR');
    alert("Failed to start streaming");
  }
});

// Stop streaming
document.getElementById("stop-btn").addEventListener("click", async () => {
  try {
    addLog('Stopping streaming...', 'INFO');
    const response = await fetch("/stop", { method: "POST" });
    const data = await response.json();

    if (data.status === "ok") {
      addLog('Streaming stopped successfully', 'INFO');
      updateStatus();
    }
  } catch (error) {
    addLog(`Failed to stop streaming: ${error.message}`, 'ERROR');
    alert("Failed to stop streaming");
  }
});

// Initialize clear log button
document.addEventListener('DOMContentLoaded', () => {
  const clearBtn = document.getElementById('clear-log-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      document.getElementById('log-container').innerHTML = '';
      addLog('Log cleared', 'INFO');
    });
  }
});

// Initialize
addLog('IRIS Client initializing...', 'INFO');
connectPreview();
connectResults();
updateStatus();
statusInterval = setInterval(updateStatus, 1000);
