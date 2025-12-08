// WebSocket for preview
let previewWs = null;
let resultsWs = null;
let statusInterval = null;

// Canvas preview state
let previewCanvas = null;
let previewCtx = null;
let canvasInitialized = false;
let lastFrameTime = 0;

// Client camera state
let clientCameraStream = null;
let clientCameraVideo = null;
let clientCameraMode = false;
let clientCaptureInterval = null;

// Results history
let resultsHistory = [];

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

// Toast notification system
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  toast.innerHTML = `
    <div class="toast-icon"></div>
    <div class="toast-message">${message}</div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-dismiss after duration
  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300); // Wait for animation
  }, duration);
}

// Update connection status
function updateConnectionStatus(elementId, status) {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = status;
    element.className = `status-indicator status-${status.toLowerCase()}`;
  }
}

// Initialize canvas for video preview
function initializeCanvas() {
  previewCanvas = document.getElementById("preview");
  if (!previewCanvas) {
    console.error("Preview canvas element not found");
    return false;
  }

  previewCtx = previewCanvas.getContext('2d', {
    alpha: false,  // No transparency needed for video
    desynchronized: true  // Allow desynchronized rendering for better performance
  });

  if (!previewCtx) {
    console.error("Failed to get 2D context from canvas");
    return false;
  }

  canvasInitialized = true;
  console.log("Canvas initialized for preview");
  return true;
}

// Render a frame to the canvas
async function renderFrameToCanvas(base64Data) {
  if (!canvasInitialized) {
    console.warn("Canvas not initialized, attempting to initialize");
    if (!initializeCanvas()) {
      return;
    }
  }

  try {
    // Convert base64 to Blob
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: 'image/jpeg' });

    // Decode image asynchronously using createImageBitmap
    const imageBitmap = await createImageBitmap(blob);

    // Resize canvas if dimensions changed (first frame or camera change)
    if (previewCanvas.width !== imageBitmap.width ||
        previewCanvas.height !== imageBitmap.height) {
      previewCanvas.width = imageBitmap.width;
      previewCanvas.height = imageBitmap.height;
      console.log(`Canvas resized to ${imageBitmap.width}x${imageBitmap.height}`);
      document.getElementById("preview-status").textContent =
        `Camera active - ${imageBitmap.width}x${imageBitmap.height}`;
    }

    // Draw the frame to canvas
    previewCtx.drawImage(imageBitmap, 0, 0);

    // Clean up bitmap to free memory
    imageBitmap.close();

    // Update frame timing (for debugging)
    const now = performance.now();
    if (lastFrameTime > 0) {
      const fps = 1000 / (now - lastFrameTime);
      // Optional: Update UI with actual render FPS
      // console.log(`Preview FPS: ${fps.toFixed(1)}`);
    }
    lastFrameTime = now;

  } catch (error) {
    console.error("Error rendering frame to canvas:", error);
    // Don't throw - just skip this frame and continue
  }
}

// Check browser support for required APIs
function checkBrowserSupport() {
  const issues = [];

  if (!window.createImageBitmap) {
    issues.push("createImageBitmap not supported");
  }

  if (!window.WebSocket) {
    issues.push("WebSocket not supported");
  }

  const testCanvas = document.createElement('canvas');
  if (!testCanvas.getContext('2d')) {
    issues.push("Canvas 2D context not supported");
  }

  if (issues.length > 0) {
    const msg = `Browser compatibility issues: ${issues.join(', ')}`;
    addLog(msg, 'ERROR');
    console.error(msg);
    alert('Your browser does not support required features for video preview. Please use a modern browser.');
    return false;
  }

  return true;
}

// Initialize client camera using getUserMedia
async function initClientCamera() {
  try {
    clientCameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }
    });

    // Create hidden video element
    clientCameraVideo = document.createElement('video');
    clientCameraVideo.srcObject = clientCameraStream;
    clientCameraVideo.autoplay = true;
    clientCameraVideo.play();

    addLog('Browser camera initialized', 'INFO');
    document.getElementById('camera-permission-status').textContent = 'Camera access granted';
    document.getElementById('preview-status').textContent = 'Using: Browser Camera';

    // Start capture loop
    startClientCameraCapture();

  } catch (error) {
    addLog(`Camera access failed: ${error.message}`, 'ERROR');
    document.getElementById('camera-permission-status').textContent = `Error: ${error.message}`;
    document.getElementById('preview-status').textContent = 'Camera access denied';
  }
}

// Capture frames from client camera
function startClientCameraCapture() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  clientCaptureInterval = setInterval(() => {
    if (!clientCameraMode || !clientCameraVideo || clientCameraVideo.readyState < 2) return;

    // Capture frame from video
    canvas.width = clientCameraVideo.videoWidth;
    canvas.height = clientCameraVideo.videoHeight;
    ctx.drawImage(clientCameraVideo, 0, 0);

    // Convert to JPEG and display
    canvas.toBlob((blob) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        renderFrameToCanvas(base64);
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.8);

  }, 100);  // 10 FPS
}

// Stop client camera
function stopClientCamera() {
  if (clientCameraStream) {
    clientCameraStream.getTracks().forEach(track => track.stop());
    clientCameraStream = null;
  }
  if (clientCameraVideo) {
    clientCameraVideo.srcObject = null;
    clientCameraVideo = null;
  }
  if (clientCaptureInterval) {
    clearInterval(clientCaptureInterval);
    clientCaptureInterval = null;
  }
  document.getElementById('preview-status').textContent = 'Camera inactive';
  addLog('Client camera stopped', 'INFO');
}

// Stop server preview WebSocket
function stopServerPreview() {
  if (previewWs) {
    previewWs.close();
    previewWs = null;
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
    renderFrameToCanvas(event.data);
    // Update status only once when first frame arrives
    if (document.getElementById('preview-status').textContent === 'Camera inactive') {
      const select = document.getElementById('server-camera-select');
      const selectedOption = select.options[select.selectedIndex];
      const cameraName = selectedOption ? selectedOption.textContent : 'Server Camera';
      document.getElementById('preview-status').textContent = `Using: ${cameraName}`;
    }
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

  // Add to history
  resultsHistory.push(data);

  // Remove placeholder on first result
  const placeholder = container.querySelector('.results-placeholder');
  if (placeholder) {
    placeholder.remove();
  }

  // Create result element
  const resultDiv = document.createElement('div');
  resultDiv.className = 'result-item';

  // Format timestamp
  const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();

  // Build result HTML
  resultDiv.innerHTML = `
    <div class="result-header">
      <span class="result-timestamp">${timestamp}</span>
      <span class="result-job-id">${data.job_id}</span>
      <span class="result-frames">Frames: ${data.frames_processed || 0}</span>
    </div>
    <div class="result-text">
      <p>${data.result || 'No result'}</p>
    </div>
    <div class="result-metrics">
      <span>Inference: ${(data.metrics?.inference_time || data.inference_time || 0).toFixed(3)}s</span>
    </div>
  `;

  // APPEND to container (not replace!)
  container.appendChild(resultDiv);

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
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

    // Update tunnel hostname field if available
    if (data.config.ssh_tunnel && data.config.ssh_tunnel.remote_host) {
      document.getElementById("tunnel-hostname").value = data.config.ssh_tunnel.remote_host;
    }

    // Update streaming server connection status
    const streamingServerStatus = data.streaming_server_status || "disconnected";
    updateConnectionStatus('streaming-server',
      streamingServerStatus.charAt(0).toUpperCase() + streamingServerStatus.slice(1));

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

  const tunnelHostname = document.getElementById("tunnel-hostname").value;

  try {
    // Update server config
    const response = await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });

    // Update tunnel hostname
    const tunnelResponse = await fetch("/tunnel/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ remote_host: tunnelHostname }),
    });

    if (response.ok && tunnelResponse.ok) {
      showToast("Configuration updated successfully", "success");
      updateStatus();
    } else {
      showToast("Configuration update partially failed", "error");
    }
  } catch (error) {
    showToast("Configuration update failed", "error");
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
      showToast('Streaming started', 'success', 3000);
      updateStatus();
    } else {
      addLog(`Start failed: ${data.message}`, 'ERROR');
      showToast(`Failed to start: ${data.message}`, "error");
    }
  } catch (error) {
    addLog(`Failed to start streaming: ${error.message}`, 'ERROR');
    showToast("Failed to start streaming", "error");
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
      showToast('Streaming stopped', 'info', 3000);
      updateStatus();
    }
  } catch (error) {
    addLog(`Failed to stop streaming: ${error.message}`, 'ERROR');
    showToast("Failed to stop streaming", "error");
  }
});

// Load available server cameras
async function loadServerCameras() {
  try {
    const response = await fetch('/cameras');
    const data = await response.json();

    const select = document.getElementById('server-camera-select');
    select.innerHTML = '';

    if (data.cameras.length === 0) {
      select.innerHTML = '<option>No cameras found</option>';
      addLog('No server cameras found', 'WARNING');
      return;
    }

    data.cameras.forEach(camera => {
      const option = document.createElement('option');
      option.value = camera.index;
      option.textContent = `Camera ${camera.index} (${camera.resolution})`;
      select.appendChild(option);
    });

    addLog(`Found ${data.cameras.length} server cameras`, 'INFO');
  } catch (error) {
    addLog(`Failed to load cameras: ${error.message}`, 'ERROR');
  }
}

// Initialize clear log button
document.addEventListener('DOMContentLoaded', () => {
  const clearBtn = document.getElementById('clear-log-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      document.getElementById('log-container').innerHTML = '';
      addLog('Log cleared', 'INFO');
    });
  }

  // Camera mode toggle handler
  document.getElementById('camera-mode').addEventListener('change', (e) => {
    const mode = e.target.value;
    clientCameraMode = (mode === 'client');

    document.getElementById('server-camera-options').style.display =
      mode === 'server' ? 'block' : 'none';
    document.getElementById('client-camera-options').style.display =
      mode === 'client' ? 'block' : 'none';

    if (mode === 'client') {
      stopServerPreview();
    } else {
      stopClientCamera();
      connectPreview();
    }
  });

  // Server camera selection handler
  document.getElementById('server-camera-select').addEventListener('change', async (e) => {
    const cameraIndex = parseInt(e.target.value);

    try {
      const response = await fetch('/camera/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: cameraIndex })
      });
      const data = await response.json();

      if (data.status === 'ok') {
        const selectedOption = e.target.options[e.target.selectedIndex];
        document.getElementById('preview-status').textContent = `Using: ${selectedOption.textContent}`;
        addLog(`Switched to camera ${cameraIndex}`, 'INFO');
      } else {
        addLog(`Failed to switch camera: ${data.message}`, 'ERROR');
      }
    } catch (error) {
      addLog(`Camera switch failed: ${error.message}`, 'ERROR');
    }
  });

  // Enable client camera button handler
  document.getElementById('enable-client-camera-btn').addEventListener('click', initClientCamera);

  // Refresh cameras button handler
  document.getElementById('refresh-cameras-btn').addEventListener('click', loadServerCameras);
});

// Initialize
addLog('IRIS Client initializing...', 'INFO');
if (checkBrowserSupport()) {
  if (!initializeCanvas()) {
    addLog('Failed to initialize canvas for preview', 'ERROR');
  }

  // Set client camera as default mode
  clientCameraMode = true;

  // Load server cameras in background (for when user switches modes)
  loadServerCameras();

  // Don't auto-connect to server preview (client mode is default)
  // User will click "Enable Camera Access" button

  connectResults();
  updateStatus();
  statusInterval = setInterval(updateStatus, 1000);
}
