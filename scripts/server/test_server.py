"""Test server to receive video streams from clients."""

import json
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="IRIS Test Server")


# Store frames for display
latest_frame = {"data": None, "timestamp": None, "frame_id": 0, "fps": 0}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve viewer page."""
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>IRIS Stream Viewer</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f0f0f0;
                }
                .container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { color: #333; }
                #video {
                    max-width: 100%;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                #stats {
                    padding: 15px;
                    background: #f9f9f9;
                    border-radius: 5px;
                    font-family: monospace;
                    margin: 10px 0;
                }
                .status {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 3px;
                    margin-left: 10px;
                }
                .connected { background: #4CAF50; color: white; }
                .disconnected { background: #f44336; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎥 IRIS Stream Viewer</h1>
                <div>
                    Status: <span id="status" class="status disconnected">Disconnected</span>
                </div>
                <img id="video" src="" alt="Waiting for stream...">
                <div id="stats">
                    <div>Frame: <span id="frame-id">-</span></div>
                    <div>FPS: <span id="fps">0.00</span></div>
                    <div>Time: <span id="time">-</span></div>
                </div>
            </div>
            <script>
                const ws = new WebSocket("ws://localhost:8000/ws/stream");
                const img = document.getElementById('video');
                const status = document.getElementById('status');
                const frameId = document.getElementById('frame-id');
                const fps = document.getElementById('fps');
                const time = document.getElementById('time');
                
                ws.onopen = function() {
                    status.textContent = 'Connected';
                    status.className = 'status connected';
                    console.log('✅ Connected to server');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    img.src = 'data:image/jpeg;base64,' + data.frame;
                    frameId.textContent = data.frame_id;
                    fps.textContent = data.fps.toFixed(2);
                    time.textContent = new Date(data.timestamp * 1000).toLocaleTimeString();
                };
                
                ws.onerror = function(error) {
                    console.error('❌ WebSocket Error:', error);
                };
                
                ws.onclose = function() {
                    status.textContent = 'Disconnected';
                    status.className = 'status disconnected';
                    console.log('🔌 Connection closed');
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Receive video stream from client."""
    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "unknown"
    print(f"✅ Client connected: {client_addr}")

    frame_count = 0
    start_time = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if start_time is None:
                start_time = datetime.now()

            frame_count += 1

            # Update latest frame
            latest_frame.update(message)

            # Log every 30 frames
            if frame_count % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"📊 Frame {message['frame_id']:4d} | "
                    f"Client FPS: {message['fps']:5.2f} | "
                    f"Server FPS: {actual_fps:5.2f}"
                )

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client_addr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if start_time:
            elapsed = (datetime.now() - start_time).total_seconds()
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📈 Session Statistics:")
            print(f"   Total frames: {frame_count}")
            print(f"   Duration: {elapsed:.2f}s")
            print(f"   Average FPS: {actual_fps:.2f}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "latest_frame_id": latest_frame["frame_id"],
        "latest_fps": latest_frame["fps"],
    }


if __name__ == "__main__":
    print("\n🚀 Starting IRIS Test Server...")
    print("=" * 50)
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/stream")
    print("🌐 Web viewer:         http://localhost:8000")
    print("💚 Health check:       http://localhost:8000/health")
    print("=" * 50)
    print("\nWaiting for connections...\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
