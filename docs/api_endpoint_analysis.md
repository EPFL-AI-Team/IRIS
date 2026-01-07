# IRIS API Endpoint Documentation & Analysis

**Last Updated:** 2026-01-05
**Status:** Complete Audit
**Purpose:** Comprehensive documentation of all HTTP and WebSocket endpoints with redundancy analysis

---

## Executive Summary
****
### Overview

The IRIS system has evolved from an **HTTP-first** to a **WebSocket-first** architecture during development. This document catalogs all endpoints, identifies redundancies, analyzes deprecated functionality, and provides recommendations for architectural cleanup.

### Key Findings

**Redundancies Identified:**
1. **Duplicate `/health` endpoints** - Server has two implementations
2. **Duplicate queue clear logic** - Client proxies server endpoint
3. **Dual entry points** - `server/app.py` (legacy) vs `server/main.py` (modern)

**Deprecations Identified:**
1. **Browser camera mode** - Removed in favor of server-side camera only
2. **Legacy start/stop HTTP endpoints** - Superseded by `/ws/client` WebSocket messages
3. **Session init endpoint** - Superseded by auto-creation on WebSocket connect
4. **Job management endpoints** - Dormant (not actively used by client backend)

### Architecture Evolution

| Phase | Design Pattern | Key Characteristics |
|-------|---------------|---------------------|
| **Original** | HTTP-first | REST endpoints for commands + WebSocket for video streaming |
| **Current** | WebSocket-first | WebSocket for real-time bidirectional + HTTP for config/queries |
| **Artifacts** | Transition debt | Legacy HTTP endpoints coexist with newer WebSocket message types |

---

## Part 1: Complete Endpoint Catalog

### 1.1 Inference Server Endpoints

#### Entry Points

The server has **two entry points** (redundant):

| File | Status | Description | Line Reference |
|------|--------|-------------|----------------|
| [src/iris/server/app.py](../src/iris/server/app.py) | Legacy | Monolithic FastAPI app with inline routes | Lines 1-1022 |
| [src/iris/server/main.py](../src/iris/server/main.py) | Modern | Modular FastAPI app using APIRouter pattern | Lines 1-190 |

#### HTTP Endpoints (Inline - server/app.py)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/config/defaults` | GET | Return server configuration defaults for frontend initialization | 304 | **Active** | Returns server, video, and segment config |
| `/api/queue/clear` | POST | Clear inference queue and free GPU memory | 346 | **Active** | Authoritative queue clear implementation |
| `/health` | GET | Health check with status, queue depth, and model loaded state | 388 | **DUPLICATE** | Should be removed (see redundancy analysis) |

**Details:**

**`GET /api/config/defaults`** (Line 304)
- **Returns:** `{server: {host, port, model_id}, video: {width, height, capture_fps, jpeg_quality, camera_index}, segment: {segment_time, frames_per_segment, overlap_frames}}`
- **Usage:** Client backend calls during initialization to populate UI defaults
- **Source:** Reads from YAML config file (`_yaml_config`)

**`POST /api/queue/clear`** (Line 346)
- **Returns:** `{cleared: int, status: "ok"}`
- **Actions:**
  1. Empties `asyncio.Queue` of pending inference jobs
  2. Calls `gc.collect()` to free Python memory
  3. Calls `torch.cuda.empty_cache()` if CUDA available
- **Usage:** Called by client backend during session reset to prevent "ghost results"
- **Critical:** Without this, old inference results arrive after user resets UI

**`GET /health`** (Line 388) **[DUPLICATE]**
- **Returns:** `{status: "ok", queue_depth: int, model_loaded: bool}`
- **Usage:** Monitoring and client polling
- **Problem:** Duplicate of `/health` in `routes/system.py` with different response schema

#### HTTP Endpoints (Router - server/routes/system.py)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/health` | GET | Health check with model loaded status | 14 | **Active** | Preferred router-based version |
| `/metrics` | GET | Detailed metrics and statistics | 24 | **Active** | Requires `enable_metrics=true` |
| `/system/clear` | DELETE | Full system reset (queue, jobs, metrics) | 42 | **Active** | More comprehensive than `/api/queue/clear` |

**Details:**

**`GET /health`** (Line 14)
- **Returns:** `{status: "healthy"|"loading", model_loaded: bool}`
- **Usage:** Health checks via `main.py` entry point
- **Note:** Missing `queue_depth` field that inline version has

**`GET /metrics`** (Line 24)
- **Returns:** `{enable_metrics: bool, stats: {...}, recent_jobs: [...]}`
- **Usage:** Observability and debugging
- **Condition:** Returns error if `config.enable_metrics` is false

**`DELETE /system/clear`** (Line 42)
- **Parameters:** `clear_logs: bool = true`, `stop_active_jobs: bool = true`
- **Returns:** `{status: "success"|"partial_success", cleared: {...}, errors: [...], timestamp: float}`
- **Actions:**
  1. Stops all active jobs (via `job_manager.stop_all_jobs()`)
  2. Clears pending jobs from queue
  3. Resets metrics and optionally deletes metric files
  4. Returns detailed status of each operation
- **Difference from `/api/queue/clear`:** Much more comprehensive, includes job lifecycle management

#### HTTP Endpoints (Router - server/routes/jobs.py)

**Status:** **DORMANT** (Not actively used by client backend)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/jobs/start` | POST | Start new job with JobConfig | 13 | Dormant | Job-based API pattern not adopted |
| `/jobs/{job_id}/stop` | DELETE | Stop running job by ID | 44 | Dormant | Alternative to WebSocket-based control |
| `/jobs/{job_id}/status` | GET | Get job status details | 75 | Dormant | Job monitoring endpoint |
| `/jobs/{job_id}/trigger` | POST | Manually trigger VideoJob inference | 94 | Dormant | Debug/testing endpoint |
| `/jobs/active` | GET | List all active jobs | 125 | Dormant | Job listing endpoint |

**Analysis:**

These endpoints represent a **"Job-based API"** design pattern where external clients could submit jobs via REST and poll for results. The client backend never adopted this pattern, instead using WebSocket streaming directly. Jobs are created internally by the server, not via these HTTP endpoints.

**Potential Use Cases:**
- External API integrations (non-WebSocket clients)
- Batch processing from scripts
- REST-first clients that don't support WebSockets

**Recommendation:** Mark as **Experimental** in API docs, document limitations, keep for potential future external integrations.

#### WebSocket Endpoints (server/app.py)

| Endpoint | Protocol | Purpose | Line | Status | Message Types |
|----------|----------|---------|------|--------|---------------|
| `/ws/stream` | WebSocket | **Primary data plane** - Frame ingestion and result streaming | 404 | **Active** | `session_config`, `session_ack`, `frame`, `result`, `session_metrics`, `complete`, `processing_complete` |
| `/ws/logs` | WebSocket | Server log streaming for debugging | 988 | **Active** | Log records (if `enable_log_streaming=true`) |

**Details:**

**`WS /ws/stream`** (Line 404) **[CRITICAL ENDPOINT]**

**Protocol Flow:**
1. Client connects and sends `session_config` message with parameters:
   ```json
   {
     "type": "session_config",
     "segment_time": 3.0,
     "frames_per_segment": 8,
     "overlap_frames": 4,
     "mode": "live" | "analysis",
     "batch_size": 16
   }
   ```

2. Server responds with `session_ack`:
   ```json
   {
     "type": "session_ack",
     "session_id": "uuid-generated-by-server"
   }
   ```

3. Client streams frames:
   ```json
   {
     "type": "frame",
     "frame_id": "uuid",
     "timestamp": 1234567890.123,
     "frame_data": "base64-encoded-jpeg"
   }
   ```

4. Server sends results as inference completes:
   ```json
   {
     "type": "result",
     "segment_id": "uuid",
     "result": {"generated_text": "..."},
     "inference_time": 1.23,
     "video_time_ms": 5000
   }
   ```

5. Server broadcasts metrics every 1 second:
   ```json
   {
     "type": "session_metrics",
     "fps": 5.2,
     "queue_depth": 3,
     "total_frames": 42,
     "total_results": 38
   }
   ```

6. **Analysis Mode Only** - Completion handshake:
   - Client sends `{"type": "complete"}` when all frames sent
   - Server flushes buffers and remaining batches
   - Server responds `{"type": "processing_complete"}` when done
   - Client closes connection

**Backpressure Mechanisms:**
- **Live Mode:** If `queue_depth >= live_queue_threshold`, server drops frames immediately
- **Analysis Mode:** Accumulates segments into batches for GPU efficiency

**`WS /ws/logs`** (Line 988)

**Protocol Flow:**
1. Server checks if log streaming enabled (`config.enable_log_streaming`)
2. If disabled, closes connection with code 1008
3. If enabled, accepts and registers connection with `WebSocketLogHandler`
4. Server broadcasts all Python `logging` records to connected clients
5. Sends `{"type": "ping"}` every 30 seconds to keep alive

**Log Format:**
```json
{
  "type": "log",
  "level": "INFO"|"WARNING"|"ERROR",
  "name": "module.name",
  "message": "Log message",
  "timestamp": 1234567890.123
}
```

**Usage:** Frontend connects to display real-time server logs in developer console

---

### 1.2 Client Backend Endpoints

All endpoints defined in [src/iris/client/web/routes.py](../src/iris/client/web/routes.py).

**Routers:**
- `api_router = APIRouter(prefix="/api")` - Line 26
- `ws_router = APIRouter(prefix="/ws")` - Line 27

#### Configuration & Status

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/status` | GET | Get client status (camera, streaming, config, FPS) | 68 | **Active** | Frontend polling |
| `/api/config` | POST | Update server configuration | 80 | **Active** | Settings panel |
| `/api/config/defaults` | GET | Get default config from YAML | 127 | **Active** | Frontend initialization |
| `/api/config/gemini-key` | GET | Check if Gemini API key configured (without exposing it) | 97 | **Active** | Report feature availability |
| `/api/config/gemini-key` | POST | Store Gemini API key for session | 113 | **Active** | Report setup |

**Details:**

**`GET /api/status`** (Line 68)
- **Returns:** `{camera: bool, streaming: bool, config: {...}, fps: float|null}`
- **Usage:** Frontend polls every 2 seconds to update status indicators
- **Source:** `AppState` in-memory state

**`POST /api/config`** (Line 80)
- **Body:** `{server?: {...}, video?: {...}, segment?: {...}}`
- **Returns:** `{status: "ok"}`
- **Actions:** Updates `state.config` in memory (not persisted to YAML)

**`GET /api/config/defaults`** (Line 127)
- **Returns:** `{server: {...}, video: {...}, segment: {...}}`
- **Source:** Reads from `config.yaml` via `ServerConfig`
- **Usage:** Frontend initialization to populate settings UI

**`GET /api/config/gemini-key`** (Line 97)
- **Returns:** `{configured: bool, source: "environment"|"stored"|"none"}`
- **Security:** Never exposes actual key value
- **Sources:** Checks `GOOGLE_API_KEY`, `GEMINI_API_KEY` env vars, or `gemini_api_key_store` global

**`POST /api/config/gemini-key`** (Line 113)
- **Body:** `{api_key: string}`
- **Returns:** `{status: "ok", message: string}`
- **Storage:** Stores in `gemini_api_key_store` global variable (in-memory, not persistent)
- **Security Note:** Storing API keys in globals is not best practice for production

#### Session Management (Legacy)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/session/init` | POST | Initialize session with config before streaming | 49 | **DEPRECATED** | Superseded by auto-creation on `/ws/client` connect |
| `/api/session/reset` | POST | Full session reset (client + inference server queue) | 1381 | **Active** | Critical for preventing ghost results |

**Details:**

**`POST /api/session/init`** (Line 49) **[DEPRECATED]**
- **Body:** `SessionConfig {frames_per_segment: int, overlap_frames: int}`
- **Returns:** `{session_id: string, status: "initialized"}`
- **Actions:** Generates UUID, stores in `session_store` dict
- **Problem:** `session_store` is partially unused, session auto-created on WS connect anyway
- **Recommendation:** Remove in v2.0 after confirming frontend migration

**`POST /api/session/reset`** (Line 1381) **[CRITICAL]**
- **Returns:** `{session_id: string (new), status: "ok"}`
- **Actions:**
  1. Generate new `session_id` via `state.reset_session()`
  2. Clear `results_history` in memory
  3. Clear database logs/results for old session
  4. **Critical:** HTTP POST to inference server `/api/queue/clear` (lines 1402-1412)
  5. Create new session in database
  6. Update `AppState` with new session
- **Why Critical:** Without step 4, old inference results (queued in GPU) arrive after reset, causing "ghost results" in new session
- **Error Handling:** Queue clear failure logs warning but allows client reset to proceed (non-fatal)

**Reset Flow:**
```
User clicks "Reset" → POST /api/session/reset
                     ↓
                  old_id = state.session_id
                  new_id = uuid4()
                     ↓
                  Clear DB (old_id)
                     ↓
                  POST → Server /api/queue/clear ← CRITICAL
                     ↓
                  Create DB session (new_id)
                     ↓
                  Return {session_id: new_id}
```

#### Camera Management (Server-Side Only)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/cameras` | GET | List available camera devices on server | 204 | **Active** | Scans indices 0-9 |
| `/api/camera/select` | POST | Switch to different camera device | 228 | **Active** | Stops current, starts new |

**Details:**

**`GET /api/cameras`** (Line 204)
- **Returns:** `{cameras: [{index: int, name: string, resolution: string}, ...]}`
- **Implementation:** Uses OpenCV `cv2.VideoCapture(i, cv2.CAP_V4L2)` to probe indices 0-9
- **Note:** V4L2 is Linux-specific; may need platform detection for cross-platform support

**`POST /api/camera/select`** (Line 228)
- **Body:** `{camera_index: int}`
- **Returns:** `{status: "ok"|"error", camera_index: int, message?: string}`
- **Actions:**
  1. Stop current camera if running (`state.camera.stop()`)
  2. Update `state.config.video.camera_index`
  3. Create new `CameraCapture` with index
  4. Start camera and verify success
- **Usage:** Camera selection dropdown in frontend

**Deprecation Note:** Comment on line 38 states `# camera_mode removed - server-only camera selection`. Originally, the system supported browser WebRTC camera streaming, but this was removed in favor of server-side capture for better control.

#### Live Streaming Control (Legacy HTTP)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/start` | POST | Start camera and streaming to inference server | 142 | **DEPRECATED** | Superseded by `/ws/client` messages |
| `/api/stop` | POST | Stop streaming and camera | 184 | **DEPRECATED** | Superseded by `/ws/client` messages |

**Details:**

**`POST /api/start`** (Line 142) **[DEPRECATED]**
- **Body:** `StartRequest {frames_per_segment?: int, overlap_frames?: int}` (optional)
- **Returns:** `{status: "ok"|"error", message: string}`
- **Actions:**
  1. Start camera (`CameraCapture`) if not running
  2. Create `StreamingClient` with config
  3. Start streaming task (`asyncio.create_task`)
  4. Store results in `state.results_history` via callback
- **Problem:** HTTP endpoint doesn't fit WebSocket-first architecture
- **New Way:** Frontend sends `{type: "start", config: {...}}` via `/ws/client`

**`POST /api/stop`** (Line 184) **[DEPRECATED]**
- **Returns:** `{status: "ok", message: "Stopped"}`
- **Actions:**
  1. Stop `StreamingClient`
  2. Stop `CameraCapture`
  3. Clear session state (`current_session`, `session_id`)
- **New Way:** Frontend sends `{type: "stop"}` via `/ws/client`

**Migration Path:**
1. Verify frontend no longer calls these endpoints (grep for `/api/start`, `/api/stop`)
2. Add HTTP 410 Gone responses with deprecation message
3. Remove in v2.0 after 2 releases

#### Results Management

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/results/history` | GET | Get all stored inference results from current session | 255 | **Active** | In-memory results |
| `/api/results/clear` | POST | Clear stored inference results history | 265 | **Active** | Cleanup |

**Details:**

**`GET /api/results/history`** (Line 255)
- **Returns:** `{count: int, results: [...]}`
- **Source:** `state.results_history` list (in-memory, max 1000 items)
- **Usage:** Results panel in live mode UI
- **Note:** Different from database results (used by analysis mode)

**`POST /api/results/clear`** (Line 265)
- **Returns:** `{status: "ok", message: "Results history cleared"}`
- **Actions:** Calls `state.results_history.clear()`
- **Scope:** Only clears in-memory list, not database

#### Video Analysis

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/datasets` | GET | List available videos and annotations in static/videos/ | 278 | **Active** | Returns metadata |
| `/api/videos/{filename}` | GET | Serve video files with range support | 327 | **Active** | HTML5 video playback |
| `/api/analysis/start` | POST | Initialize video analysis job | 436 | **Active** | Prepares resources |
| `/api/analysis/stop` | POST | Stop ongoing analysis | 657 | **Active** | Cleanup |

**Details:**

**`GET /api/datasets`** (Line 278)
- **Returns:** `{videos: [...], annotations: [...]}`
- **Video Metadata:** `{filename, path, size_mb, duration_sec, resolution, fps, frame_count}`
- **Annotation Metadata:** `{filename, path, size_kb, line_count}`
- **Implementation:**
  - Glob `*.mp4` files in `static/videos/`
  - Use OpenCV to read video metadata
  - Glob `*.jsonl` files for annotations
- **Usage:** Dataset browser in analysis mode UI

**`GET /api/videos/{filename}`** (Line 327)
- **Returns:** `FileResponse` with `video/mp4` content type
- **Headers:** `Accept-Ranges: bytes` (enables seeking)
- **Security:** Checks file is within `static/videos/` directory (prevents path traversal)
- **Usage:** HTML5 `<video>` element src attribute

**`POST /api/analysis/start`** (Line 436)
- **Body:** `{video_filename: str, annotation_filename?: str, segment_time: float, frames_per_segment: int, overlap_frames: int, simulation_fps?: float (deprecated)}`
- **Returns:** `{job_id: string, video_file: string, total_frames: int, duration_sec: float, simulation_fps: float, config: {...}}`
- **Actions:**
  1. Validate video file exists
  2. Clear inference server queue (POST `/api/queue/clear`) **[CRITICAL]**
  3. Parse annotations if provided (OpenAI Chat format JSONL)
  4. Create `VideoFileCapture` with deterministic seeking
  5. Calculate simulation FPS and frame indices
  6. Store job metadata in `state.active_analysis_job`
  7. Create session in database
- **Why Queue Clear is Critical:** Prevents old live mode results from mixing with analysis results
- **Note:** Uses deprecated `simulation_fps` parameter for backward compatibility

**`POST /api/analysis/stop`** (Line 657)
- **Returns:** `{status: "ok", message: "Analysis stopped"}`
- **Actions:**
  1. Stop `StreamingClient`
  2. Cancel streaming task
  3. Stop `VideoFileCapture`
  4. Clear `active_analysis_job`
- **Usage:** User cancellation or error recovery

#### Database Session Management

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/sessions` | GET | List all analysis sessions (last 50) | 1076 | **Active** | Session browser |
| `/api/sessions/{session_id}` | GET | Get specific session details by ID | 1085 | **Active** | Session restoration |
| `/api/sessions/{session_id}` | DELETE | Delete session and all related data | 1132 | **Active** | Cleanup |
| `/api/sessions/{session_id}/results` | GET | Get all results for session | 1096 | **Active** | Report generation |
| `/api/sessions/{session_id}/results` | DELETE | Clear all results for session | 1123 | **Active** | Cleanup |
| `/api/sessions/{session_id}/logs` | GET | Get session logs (limit: 1000) | 1105 | **Active** | Debugging |
| `/api/sessions/{session_id}/logs` | DELETE | Clear all logs for session | 1114 | **Active** | Cleanup |
| `/api/session/{session_id}/data` | GET | Get complete session data for restoration | 1342 | **Active** | Session recovery |

**Details:**

All these endpoints interact with SQLite database via repository pattern:
- `session_repo` - Sessions table
- `results_repo` - Inference results table
- `logs_repo` - Log entries table

**Database Schema (Inferred):**

**sessions:**
- `id` (session_id)
- `status` (created, running, completed, error)
- `created_at`
- `config` (JSON)
- `video_file`
- `annotation_file`

**results:**
- `session_id` (FK)
- `job_id`
- `video_time_ms`
- `inference_start_ms`
- `inference_end_ms`
- `frame_start`
- `frame_end`
- `result` (JSON)

**logs:**
- `session_id` (FK)
- `timestamp`
- `level` (INFO, WARNING, ERROR)
- `message`

#### Report Generation

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/report/generate` | POST | Generate Gemini-powered report with streaming | 1146 | **Active** | Markdown streaming |
| `/api/report/{session_id}` | GET | Get latest stored report for session | 1323 | **Active** | Report retrieval |
| `/api/report/fallback/{session_id}` | GET | Get basic stats report (no LLM) | 1289 | **Active** | Fallback without API key |

**Details:**

**`POST /api/report/generate`** (Line 1146)
- **Body:** `{session_id: string, force_regenerate?: bool}`
- **Returns:** `StreamingResponse` with `text/markdown` content type
- **Headers:** `X-Report-Provider: gemini`
- **Actions:**
  1. Check for cached report (unless `force_regenerate=true`)
  2. Fetch session from database
  3. If not in DB, use in-memory `AppState` (live sessions)
  4. Fetch results, logs, annotations
  5. Check for Gemini API key (env vars or `gemini_api_key_store`)
  6. If no key, return fallback report
  7. Stream report chunks via `generate_report_stream()`
  8. Store complete report in database (if session in DB)
- **Report Content:** Includes session overview, results analysis, performance metrics, insights
- **Gemini Model:** Uses `google-generativeai` SDK
- **Usage:** "Generate Report" button in frontend

**`GET /api/report/{session_id}`** (Line 1323)
- **Returns:** `{session_id: string, report: {content: string, provider: string, created_at: timestamp}} | {error: string}`
- **Source:** `reports_repo.get_latest_by_session()`
- **Usage:** Retrieve previously generated report without regenerating

**`GET /api/report/fallback/{session_id}`** (Line 1289)
- **Returns:** `{report: string (markdown), provider: "fallback"}`
- **Content:** Basic statistics without LLM:
  - Total inference count
  - Duration
  - Average inference time
  - Detected actions (if annotations present)
- **Usage:** When no Gemini API key configured

#### Queue Management (Proxy)

| Endpoint | Method | Purpose | Line | Status | Notes |
|----------|--------|---------|------|--------|-------|
| `/api/queue/clear` | POST | Proxy to inference server queue clear | 1437 | **Active** | Adds indirection |

**Details:**

**`POST /api/queue/clear`** (Line 1437) **[PROXY]**
- **Returns:** `{status: "ok"|"error", cleared?: int, message?: string}`
- **Actions:**
  1. Use `httpx.AsyncClient` to POST to inference server
  2. URL: `http://{server_host}:{server_port}/api/queue/clear`
  3. Timeout: 5 seconds
  4. Return proxied response or error
- **Usage:** Called by session reset and analysis start
- **Analysis:** Creates indirection layer; frontend could call server directly
- **Justification:** Provides abstraction if server URL changes, keeps frontend agnostic to backend topology
- **Recommendation:** Keep for now (low cost), but consider removing if frontend becomes more sophisticated

#### WebSocket Endpoints (client/web/routes.py)

| Endpoint | Protocol | Purpose | Line | Status | Message Types |
|----------|----------|---------|------|--------|---------------|
| `/ws/client` | WebSocket | **Unified control plane** - Preview, commands, results | 1461 | **Active** | `start`, `stop`, `clear_queue`, `reset_session`, `session_info`, `preview_frame`, `result`, `metrics`, `server_status`, `error`, `log` |
| `/ws/analysis` | WebSocket | **Analysis progress stream** - Video analysis workflow | 679 | **Active** | `progress`, `upload_complete`, `complete`, `result`, `session_ack`, `session_metrics`, `error` |

**Details:**

**`WS /ws/client`** (Line 1461) **[UNIFIED CONTROL PLANE]**

**Purpose:** Single WebSocket endpoint for all frontend communication in the new architecture.

**Protocol Flow:**
1. Client connects
2. Server sends `session_info`:
   ```json
   {
     "type": "session_info",
     "session_id": "uuid",
     "config": {...}
   }
   ```
3. Server starts background tasks:
   - `stream_preview_frames()` - Continuous USB camera preview at 10 FPS
   - `health_check_loop()` - Server health status every 5 seconds
4. Server enters message loop, handling incoming commands

**Incoming Messages (Frontend → Client Backend):**

1. **Start Inference:**
   ```json
   {
     "type": "start",
     "config": {
       "frames_per_segment": 8,
       "overlap_frames": 4
     }
   }
   ```
   **Action:** Start camera, create `StreamingClient`, connect to inference server

2. **Stop Inference:**
   ```json
   {
     "type": "stop"
     }
   ```
   **Action:** Stop streaming, keep camera running for preview

3. **Clear Queue:**
   ```json
   {
     "type": "clear_queue"
   }
   ```
   **Action:** Proxy to inference server `/api/queue/clear`

4. **Reset Session:**
   ```json
   {
     "type": "reset_session"
   }
   ```
   **Action:** Generate new session_id, clear state, clear server queue, update DB

**Outgoing Messages (Client Backend → Frontend):**

1. **Preview Frame:**
   ```json
   {
     "type": "preview_frame",
     "frame_data": "base64-jpeg",
     "timestamp": 1234567890.123
   }
   ```
   **Frequency:** 10 FPS (100ms delay between frames)

2. **Server Status:**
   ```json
   {
     "type": "server_status",
     "status": "ok"|"error",
     "queue_depth": 3
   }
   ```
   **Frequency:** Every 5 seconds

3. **Result:**
   ```json
   {
     "type": "result",
     "result": {...},
     "inference_time": 1.23,
     "video_time_ms": 5000
   }
   ```
   **Source:** Forwarded from inference server `/ws/stream`

4. **Session Metrics:**
   ```json
   {
     "type": "session_metrics",
     "fps": 5.2,
     "queue_depth": 3,
     "total_frames": 42,
     "total_results": 38
   }
   ```
   **Source:** Forwarded from inference server `/ws/stream`

**Connection Lifecycle:**
- **Connect:** Auto-create session in DB if not exists
- **Disconnect:** Stop preview, log event, keep session data
- **Reconnect:** Restore from DB using session_id from cookie/localStorage

**`WS /ws/analysis`** (Line 679) **[ANALYSIS PROGRESS STREAM]**

**Purpose:** Specialized WebSocket for video file analysis streaming with progress updates.

**Protocol Flow:**
1. Client calls `POST /api/analysis/start` first (initializes resources)
2. Client connects to `/ws/analysis`
3. Server verifies `state.analysis_video_capture` exists (else error)
4. Server connects to inference server `/ws/stream`
5. Server sends `session_config` to inference server
6. Server receives `session_ack` from inference server, forwards to frontend
7. Server streams video frames with deterministic seeking:
   ```python
   for physical_index in frame_indices:
       cap.set(CV_CAP_PROP_POS_FRAMES, physical_index)
       frame = cap.read()
       send_to_inference_server(frame)
   ```
8. Server sends throttled progress updates (max 10/sec):
   ```json
   {
     "type": "progress",
     "frames_sent": 42,
     "total_frames": 1000,
     "position_sec": 5.2,
     "duration_sec": 60.0,
     "eta_sec": 12.3,
     "fps": 8.5
   }
   ```
9. Server forwards inference results to frontend (augmented with metadata)
10. Server sends `{"type": "complete"}` to inference server when done
11. Server waits for `{"type": "processing_complete"}` from inference server
12. Server sends `{"type": "complete"}` to frontend
13. Connection closes

**Turbo Mode:** No sleep between frames; sends as fast as network allows.

**Result Augmentation:** Server adds `frame_range`, `video_time_ms`, stores to DB.

**Handshake Protocol (Critical):**
- **Without handshake:** Client disconnects before server finishes processing last batch, results lost
- **With handshake:** Client waits for explicit `processing_complete` signal before closing

---

## Part 2: Redundancy Analysis

### REDUNDANCY 1: Health Check Endpoints **[HIGH PRIORITY]**

**Location 1:** [src/iris/server/app.py:388](../src/iris/server/app.py#L388)
```python
@app.get("/health")
async def health_check() -> dict:
    state = get_server_state()
    queue_depth = state.queue.queue.qsize() if state.queue and state.queue.queue else 0
    return {
        "status": "ok",
        "queue_depth": queue_depth,
        "model_loaded": state.model_loaded,
    }
```
**Response:** `{status: "ok", queue_depth: int, model_loaded: bool}`

**Location 2:** [src/iris/server/routes/system.py:14](../src/iris/server/routes/system.py#L14)
```python
@router.get("/health")
async def health() -> dict[str, str | bool]:
    state = get_server_state()
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
    }
```
**Response:** `{status: "healthy"|"loading", model_loaded: bool}`

**Root Cause:**
- `app.py` is the legacy monolithic entry point with inline routes
- `main.py` is the modern modular entry point that includes router-based routes
- Both can run independently, causing duplicate route registration if both are active
- During refactoring from monolithic to modular, duplicate routes were not removed

**Comparison:**

| Field | app.py version | routes/system.py version |
|-------|---------------|-------------------------|
| `status` | Always `"ok"` | `"healthy"` or `"loading"` depending on model |
| `queue_depth` | ✅ Included | ❌ Missing |
| `model_loaded` | ✅ Included | ✅ Included |

**Impact:**
- **Low:** Both work, but confusing for API consumers
- **Risk:** If both entry points run simultaneously, route conflict (FastAPI would warn)
- **Maintenance:** Developers must remember to update both

**Recommendation:**

**Action:** Remove duplicate from `app.py`, enhance router version

**Step 1:** Enhance router version to include `queue_depth`:
```python
# src/iris/server/routes/system.py:14
@router.get("/health")
async def health() -> dict[str, str | bool | int]:
    state = get_server_state()
    queue_depth = state.queue.queue.qsize() if state.queue and state.queue.queue else 0
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
        "queue_depth": queue_depth,
    }
```

**Step 2:** Remove from `app.py`:
```python
# src/iris/server/app.py:388 - DELETE THIS ENDPOINT
```

**Step 3:** Update all callers to use consistent response format

**Timeline:** Immediate (v1.x patch)

---

### REDUNDANCY 2: Queue Clear Endpoints **[MEDIUM PRIORITY]**

**Location 1 (Authoritative):** [src/iris/server/app.py:346](../src/iris/server/app.py#L346)
```python
@app.post("/api/queue/clear")
async def clear_queue() -> dict:
    # Clear queue, run GC, clear GPU cache
    return {"cleared": cleared_count, "status": "ok"}
```

**Location 2 (Proxy):** [src/iris/client/web/routes.py:1437](../src/iris/client/web/routes.py#L1437)
```python
@api_router.post("/queue/clear")
async def clear_inference_queue() -> dict[str, Any]:
    # HTTP POST to inference server /api/queue/clear
    return {"status": "ok", "cleared": data.get("cleared", 0)}
```

**Data Flow:**
```
Frontend → [Option A] Client Backend /api/queue/clear → HTTP → Server /api/queue/clear

         [Option B] Direct call → Server /api/queue/clear
```

**Root Cause:**
- Client backend needs to trigger queue clear during session reset
- Instead of requiring frontend to know server URL, client provides proxy endpoint
- Adds indirection layer for convenience

**Analysis:**

**Pros of Proxy:**
- Frontend agnostic to inference server URL
- Single configuration point (client knows server URL)
- Abstraction allows changing backend topology without frontend changes

**Cons of Proxy:**
- Adds indirection (harder to debug)
- Extra network hop (negligible latency)
- Duplicates error handling logic

**Alternative Architectures:**

**Option A (Current):** Client proxy
- Frontend → Client `/api/queue/clear` → Server `/api/queue/clear`

**Option B (Direct):** Frontend calls server
- Frontend → Server `/api/queue/clear`
- Requires frontend to know server URL (already has it in config)

**Option C (Server-side):** New unified reset endpoint
- Server: `POST /api/reset` - Atomic operation that clears queue, resets metrics, returns new state token
- Client calls this during session reset
- Eliminates proxy, provides atomic transaction

**Recommendation:**

**Action:** Keep current proxy for now, document clearly

**Reasoning:**
- Low maintenance cost
- Provides useful abstraction
- Client coordination logic (session reset) already complex, keeping queue clear here is consistent
- Breaking change not justified for minimal benefit

**Documentation:**
```markdown
### Client Backend `/api/queue/clear`
**Type:** Proxy endpoint
**Proxies to:** Inference server `/api/queue/clear`
**Purpose:** Convenience wrapper for frontend to clear inference queue without knowing server URL
**Usage:** Called during session reset to prevent ghost results
```

**Future (v3.0):** If frontend becomes more sophisticated (e.g., direct server connection for observability), consider removing proxy and having frontend call server directly.

---

### REDUNDANCY 3: Dual Server Entry Points **[HIGH PRIORITY]**

**Location 1 (Legacy):** [src/iris/server/app.py](../src/iris/server/app.py)
- **Type:** Monolithic FastAPI app with inline routes
- **Lines:** 1-1022
- **Routes:** Inline `@app.get()`, `@app.post()`, `@app.websocket()`
- **Lifespan:** Inline `@asynccontextmanager` (lines 231-298)

**Location 2 (Modern):** [src/iris/server/main.py](../src/iris/server/main.py)
- **Type:** Modular FastAPI app using APIRouter pattern
- **Lines:** 1-190
- **Routes:** Imported from `routes/jobs.py`, `routes/system.py`
- **Lifespan:** Uses `LifecycleHandler` class (line 21)

**Root Cause:**
- Original implementation: Everything in `app.py` (monolithic)
- Refactoring effort: Created `main.py` with modular router architecture
- Migration incomplete: `app.py` not removed, still contains unique endpoints
- Both entry points can run independently

**Which Entry Point is Used?**

Check startup scripts:
- `python -m iris.server.app` → Runs `app.py`
- `python -m iris.server.main` → Runs `main.py`

**Unique Endpoints in app.py:**
1. `GET /api/config/defaults` (line 304)
2. `POST /api/queue/clear` (line 346)
3. `GET /health` (line 388) - **DUPLICATE**
4. `WS /ws/stream` (line 404) - **CRITICAL**
5. `WS /ws/logs` (line 988) - **CRITICAL**

**Unique Routers in main.py:**
1. `routes/jobs.py` - Job management endpoints
2. `routes/system.py` - System management endpoints

**Migration Path:**

**Step 1:** Move inline routes from `app.py` to routers

**Create `src/iris/server/routes/config.py`:**
```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/api/config/defaults")
async def get_config_defaults() -> dict:
    # Migrate code from app.py:304
    ...
```

**Create `src/iris/server/routes/websocket.py`:**
```python
from fastapi import APIRouter, WebSocket
router = APIRouter()

@router.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    # Migrate code from app.py:404
    ...

@router.websocket("/ws/logs")
async def log_streaming_endpoint(websocket: WebSocket) -> None:
    # Migrate code from app.py:988
    ...
```

**Absorb queue clear into `routes/system.py`:**
```python
# routes/system.py
@router.post("/api/queue/clear")
async def clear_queue() -> dict:
    # Migrate code from app.py:346
    # Alternative: Consolidate with /system/clear
    ...
```

**Step 2:** Update `main.py` to include new routers
```python
from iris.server.routes import jobs, system, config, websocket

app.include_router(jobs.router)
app.include_router(system.router)
app.include_router(config.router)
app.include_router(websocket.router)
```

**Step 3:** Update startup scripts to use `main.py` exclusively

**Step 4:** Remove `app.py` entirely

**Benefits:**
- Single source of truth
- Better code organization (routes grouped by functionality)
- Easier to test individual routers
- Clearer architectural boundaries
- Reduced confusion for new developers

**Timeline:** Short-term (v2.0)

---

### REDUNDANCY 4: Session Store Underutilization **[LOW PRIORITY]**

**Location:** [src/iris/client/web/routes.py:41-43](../src/iris/client/web/routes.py#L41-L43)
```python
# Global store for active sessions (Simple in-memory for demo)
# Map session_id -> {"config": dict, "created_at": float}
session_store: dict[str, dict] = {}
```

**Usage:**
- Populated by `POST /api/session/init` (line 49-58) **[DEPRECATED ENDPOINT]**
- Never read elsewhere in codebase

**Analysis:**
- Originally designed for session tracking before WebSocket connections
- Sessions now auto-created on `/ws/client` connect and stored in database
- `session_store` dict is vestigial data structure

**Recommendation:**

**Action:** Remove `session_store` when removing `/api/session/init` endpoint

**Step 1:** Verify `session_store` is only used by `/api/session/init`
```bash
git grep "session_store" src/
```

**Step 2:** Remove both in same commit:
```python
# DELETE:
# - Line 41-43: session_store definition
# - Line 49-66: /api/session/init endpoint
```

**Timeline:** v2.0 (with other deprecation removals)

---

## Part 3: Communication Flow Diagrams

### 3.1 Live Streaming Flow (Current WebSocket-First Architecture)

```
┌─────────────────┐                 ┌─────────────────┐                 ┌─────────────────┐
│     Browser     │                 │  Client Backend │                 │ Inference Server│
│   (React UI)    │                 │   (FastAPI)     │                 │   (FastAPI)     │
└────────┬────────┘                 └────────┬────────┘                 └────────┬────────┘
         │                                   │                                   │
         │  1. WS Connect /ws/client         │                                   │
         ├──────────────────────────────────>│                                   │
         │                                   │  Initialize camera                │
         │                                   │  (USB via OpenCV)                 │
         │  2. {type: "session_info"}        │                                   │
         │<──────────────────────────────────┤                                   │
         │  {session_id, config}             │                                   │
         │                                   │                                   │
         │  3. {type: "preview_frame"}       │                                   │
         │<──────────────────────────────────┤ Background task:                  │
         │  (10 FPS, continuous)             │ stream_preview_frames()           │
         │  Base64 JPEG                      │                                   │
         │                                   │                                   │
         │  4. {type: "server_status"}       │                                   │
         │<──────────────────────────────────┤ Background task:                  │
         │  (Every 5 seconds)                │ health_check_loop()               │
         │  {status, queue_depth}            │                                   │
         │                                   │                                   │
         │  5. {type: "start", config}       │                                   │
         ├──────────────────────────────────>│                                   │
         │                                   │  Create StreamingClient           │
         │                                   │                                   │
         │                                   │  6. WS Connect /ws/stream         │
         │                                   ├──────────────────────────────────>│
         │                                   │                                   │
         │                                   │  7. {type: "session_config"}      │
         │                                   ├──────────────────────────────────>│
         │                                   │  {frames_per_segment,             │
         │                                   │   overlap_frames, mode: "live"}   │
         │                                   │                                   │
         │                                   │  8. {type: "session_ack"}         │
         │                                   │<──────────────────────────────────┤
         │                                   │  {session_id: "server-uuid"}      │
         │                                   │                                   │
         │                                   │  9. {type: "frame", frame_data}   │
         │                                   ├──────────────────────────────────>│
         │                                   │  (Controlled by segment FPS)      │
         │                                   │  (e.g., 8 frames / 3 sec = 2.67fps)│
         │                                   │                                   │
         │                                   │                                   │  FrameBuffer
         │                                   │                                   │  → Segment (8 frames)
         │                                   │                                   │  → Enqueue Job
         │                                   │                                   │  → GPU Inference
         │                                   │                                   │
         │                                   │  10. {type: "session_metrics"}    │
         │  11. {type: "metrics"}            │<──────────────────────────────────┤
         │<──────────────────────────────────┤  (Every 1 second from server)     │
         │  (Forwarded)                      │  {fps, queue_depth, total_frames} │
         │                                   │                                   │
         │                                   │                                   │
         │                                   │  12. {type: "result"}             │
         │  13. {type: "result"}             │<──────────────────────────────────┤
         │<──────────────────────────────────┤  (Inference complete)             │
         │  (Forwarded + stored)             │  {result, inference_time}         │
         │                                   │                                   │
         │  14. {type: "stop"}               │                                   │
         ├──────────────────────────────────>│                                   │
         │                                   │  Close /ws/stream                 │
         │                                   ├──────────────────────────────────>│
         │                                   │  (Camera keeps running)           │
         │                                   │                                   │
         │  (Preview continues)              │                                   │
         │<──────────────────────────────────┤                                   │
```

**Key Properties:**
1. **Preview Decoupled:** Preview frames stream independently of inference
2. **Dual WebSocket:** `/ws/client` (control plane) and `/ws/stream` (data plane) are separate
3. **FPS Control:** Inference frame rate calculated from segment config: `FPS = frames_per_segment / segment_time`
4. **Result Forwarding:** Client backend receives results from server, stores in memory, forwards to frontend
5. **Metrics Passthrough:** Server metrics forwarded to frontend every 1 second
6. **Camera Persistence:** Camera continues running for preview after stopping inference

---

### 3.2 Video Analysis Flow (Turbo Mode with Batch Inference)

```
┌─────────────────┐                 ┌─────────────────┐                 ┌─────────────────┐
│     Browser     │                 │  Client Backend │                 │ Inference Server│
│   (React UI)    │                 │   (FastAPI)     │                 │   (FastAPI)     │
└────────┬────────┘                 └────────┬────────┘                 └────────┬────────┘
         │                                   │                                   │
         │  1. POST /api/analysis/start      │                                   │
         ├──────────────────────────────────>│  Validate video file exists       │
         │  {video_filename, segment_time,   │                                   │
         │   frames_per_segment,             │  *** CRITICAL STEP ***           │
         │   overlap_frames}                 │  POST /api/queue/clear            │
         │                                   ├──────────────────────────────────>│
         │                                   │                                   │  Clear queue
         │                                   │  {cleared: N}                     │  (Prevent ghost results)
         │                                   │<──────────────────────────────────┤
         │                                   │                                   │
         │                                   │  Parse annotations (if provided)  │
         │                                   │  Create VideoFileCapture          │
         │                                   │  Calculate frame indices          │
         │                                   │  Create session in DB             │
         │                                   │                                   │
         │  {job_id, total_frames,           │                                   │
         │   duration_sec, simulation_fps}   │                                   │
         │<──────────────────────────────────┤                                   │
         │                                   │                                   │
         │  2. WS Connect /ws/analysis       │                                   │
         ├──────────────────────────────────>│  Verify job ready                 │
         │                                   │                                   │
         │                                   │  3. WS Connect /ws/stream         │
         │                                   ├──────────────────────────────────>│
         │                                   │                                   │
         │                                   │  4. {type: "session_config",      │
         │                                   │      mode: "analysis",            │
         │                                   │      batch_size: 16}              │
         │                                   ├──────────────────────────────────>│
         │                                   │                                   │
         │                                   │  5. {type: "session_ack"}         │
         │  6. {type: "session_ack"}         │<──────────────────────────────────┤
         │<──────────────────────────────────┤  (Forwarded to frontend)          │
         │                                   │                                   │
         │                                   │  *** TURBO MODE LOOP ***          │
         │                                   │  for idx in physical_indices:     │
         │                                   │    cap.set(POS_FRAMES, idx)       │
         │                                   │    frame = cap.read()             │
         │                                   │    send_frame(frame)              │
         │                                   │    # NO SLEEP!                    │
         │                                   │                                   │
         │                                   │  7. {type: "frame"}               │
         │                                   ├─────────────────────────────────>│
         │                                   │  (Deterministic seeking,          │
         │                                   │   as fast as network allows)      │
         │                                   │                                   │
         │  8. {type: "progress"}            │                                   │
         │<──────────────────────────────────┤  Throttled: max 10 updates/sec    │
         │  {frames_sent, total_frames,      │                                   │
         │   position_sec, eta_sec, fps}     │                                   │
         │                                   │                                   │
         │                                   │                                   │  *** SERVER SIDE ***
         │                                   │                                   │  FrameBuffer
         │                                   │                                   │  → Segment (when buffer full)
         │                                   │                                   │  → BatchAccumulator
         │                                   │                                   │  → Batch (16 segments)
         │                                   │                                   │  → Single GPU call
         │                                   │                                   │  → Scatter results
         │                                   │                                   │
         │                                   │  9. {type: "result"}              │
         │  10. {type: "result"}             │<──────────────────────────────────┤
         │<──────────────────────────────────┤  (Multiple results in parallel)   │
         │  (Augmented with frame_range,     │                                   │
         │   stored in DB)                   │                                   │
         │                                   │                                   │
         │  11. {type: "upload_complete"}    │  End of video reached             │
         │<──────────────────────────────────┤                                   │
         │                                   │                                   │
         │                                   │  12. {type: "complete"}           │
         │                                   ├──────────────────────────────────>│
         │                                   │  *** HANDSHAKE START ***          │
         │                                   │                                   │  Flush FrameBuffer
         │                                   │                                   │  (partial segments)
         │                                   │                                   │
         │                                   │                                   │  Flush BatchAccumulator
         │                                   │                                   │  (partial batches)
         │                                   │                                   │
         │                                   │                                   │  Wait for GPU jobs
         │                                   │                                   │
         │                                   │  13. {type: "processing_complete"}│
         │  14. {type: "complete"}           │<──────────────────────────────────┤
         │<──────────────────────────────────┤  *** HANDSHAKE END ***            │
         │                                   │                                   │
         │                                   │  Update DB: status="completed"    │
         │                                   │                                   │
         │                                   │  Close /ws/stream                 │
         │                                   ├──────────────────────────────────>│
         │  (Close /ws/analysis)             │                                   │
         │                                   │                                   │
```

**Key Differences from Live Mode:**
1. **No Sleep:** Frames sent as fast as network/server can handle (10-100x real-time)
2. **Deterministic Seeking:** Physical frame indices pre-calculated, OpenCV `seek()` used
3. **Batch Inference:** Server accumulates 16 segments, processes in single GPU pass (10x throughput)
4. **Handshake Protocol:** Explicit `complete` → `processing_complete` ensures all results delivered
5. **Progress Throttling:** UI updates limited to 10/sec despite fast frame sending (prevents UI lag)
6. **Queue Clear:** Critical step before starting to prevent old results mixing with new analysis
7. **Result Augmentation:** Client adds `frame_range` and `video_time_ms` before storing in DB

**Without Handshake:**
- Client sends last frame → closes connection immediately
- Server still processing final batch (e.g., 10 segments in GPU)
- Results lost, analysis incomplete

**With Handshake:**
- Client signals `complete` after last frame
- Server flushes buffers, waits for all GPU jobs
- Server signals `processing_complete` when truly done
- Client closes connection knowing all results delivered

---

### 3.3 Session Reset Flow (Preventing Ghost Results)

```
┌─────────────────┐                 ┌─────────────────┐                 ┌─────────────────┐
│     Browser     │                 │  Client Backend │                 │ Inference Server│
│   (React UI)    │                 │   (FastAPI)     │                 │   (FastAPI)     │
└────────┬────────┘                 └────────┬────────┘                 └────────┬────────┘
         │                                   │                                   │
         │  User clicks "Reset Session"      │                                   │
         │                                   │                                   │
         │  1. POST /api/session/reset       │                                   │
         ├──────────────────────────────────>│                                   │
         │                                   │                                   │
         │                                   │  Step 1: Generate new session_id  │
         │                                   │  old_id = state.session_id        │
         │                                   │  new_id = uuid4()                 │
         │                                   │  state.session_id = new_id        │
         │                                   │                                   │
         │                                   │  Step 2: Clear in-memory state    │
         │                                   │  state.results_history.clear()    │
         │                                   │  state.camera = None              │
         │                                   │  state.streaming_client = None    │
         │                                   │                                   │
         │                                   │  Step 3: Clear database           │
         │                                   │  logs_repo.clear_logs(old_id)     │
         │                                   │  results_repo.clear_results(old_id)│
         │                                   │                                   │
         │                                   │  *** CRITICAL STEP 4 ***          │
         │                                   │  POST /api/queue/clear            │
         │                                   │  (httpx.AsyncClient, 5s timeout)  │
         │                                   ├──────────────────────────────────>│
         │                                   │                                   │
         │                                   │                                   │  while !queue.empty():
         │                                   │                                   │    job = queue.get_nowait()
         │                                   │                                   │    # Discard job
         │                                   │                                   │
         │                                   │                                   │  gc.collect()
         │                                   │                                   │  torch.cuda.empty_cache()
         │                                   │                                   │
         │                                   │  {cleared: 5, status: "ok"}       │
         │                                   │<──────────────────────────────────┤
         │                                   │                                   │
         │                                   │  Step 5: Create new session       │
         │                                   │  session_repo.create(             │
         │                                   │    session_id=new_id,             │
         │                                   │    config=state.session_config    │
         │                                   │  )                                │
         │                                   │                                   │
         │  2. {session_id: new_id,          │                                   │
         │      status: "ok"}                │                                   │
         │<──────────────────────────────────┤                                   │
         │                                   │                                   │
         │  UI updates:                      │                                   │
         │  - Display new_id                 │                                   │
         │  - Clear results panel            │                                   │
         │  - Reset metrics to 0             │                                   │
```

**Why Queue Clear is Critical:**

**Without Queue Clear (The Ghost Results Problem):**
```
Time 0: User streaming, 10 frames in inference queue
Time 1: User clicks "Reset"
Time 2: Client generates new session_id, clears UI
Time 3: Inference server still processing old frames from queue
Time 4-10: Old results arrive with outdated session context
Time 11: Frontend receives results, displays in new session
Result: User sees "ghost results" from previous session
```

**With Queue Clear:**
```
Time 0: User streaming, 10 frames in inference queue
Time 1: User clicks "Reset"
Time 2: Client calls server /api/queue/clear
Time 3: Server empties queue (10 jobs discarded)
Time 4: Client generates new session_id, clears UI
Time 5+: No old results arrive
Result: Clean slate, no ghost results
```

**Error Handling:**

```python
# routes.py:1402-1412
try:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"http://{host}:{port}/api/queue/clear")
        if response.status == 200:
            logger.info(f"Cleared inference queue: {data}")
        else:
            logger.warning(f"Failed to clear queue: {response.status}")
except Exception as e:
    logger.error(f"Failed to clear inference queue: {e}")
    # ALLOW CLIENT RESET TO PROCEED (non-fatal)
```

**Design Decision:** Queue clear failure is non-fatal
- **Rationale:** Client reset should succeed even if server unreachable
- **Trade-off:** May have ghost results if server was unreachable but comes back online
- **Mitigation:** Server-side session expiration (not yet implemented)

---

## Part 4: HTTP vs WebSocket Separation

### Design Principle

| Protocol | Use Cases | Characteristics |
|----------|-----------|----------------|
| **HTTP REST** | Configuration, queries, initialization, one-time operations | Stateless, request-response, cacheable |
| **WebSocket** | Streaming, real-time updates, bidirectional events | Stateful, persistent connection, low latency |

### Current Separation Analysis

#### HTTP Endpoints (Should be stateless, one-time operations)

**✅ Good Examples:**

| Endpoint | Justification |
|----------|---------------|
| `GET /api/config/defaults` | One-time query for initialization |
| `GET /api/datasets` | List static resources |
| `POST /api/analysis/start` | Initialize analysis job (prepare resources) |
| `GET /api/sessions` | Query historical data |
| `POST /api/report/generate` | One-time operation with streaming response |
| `DELETE /api/sessions/{id}` | One-time delete operation |

**❌ Violations (Transitional):**

| Endpoint | Issue | Status | New Way |
|----------|-------|--------|---------|
| `POST /api/start` | Starts persistent streaming operation | **DEPRECATED** | `WS /ws/client` message `{type: "start"}` |
| `POST /api/stop` | Stops persistent streaming operation | **DEPRECATED** | `WS /ws/client` message `{type: "stop"}` |
| `POST /api/session/init` | Session lifecycle should be WebSocket-managed | **DEPRECATED** | Auto-create on `WS /ws/client` connect |

**Justification for Deprecation:**

**Problem with HTTP for streaming control:**
- HTTP is stateless, but streaming requires persistent state
- Start/stop should be tied to WebSocket connection lifecycle
- HTTP endpoints don't detect client disconnect (can't auto-cleanup)
- Leads to resource leaks (camera, streaming client still running after disconnect)

**WebSocket Solution:**
- Connection lifecycle tied to resource lifecycle
- Automatic cleanup on disconnect
- Bidirectional: can send status updates proactively
- Natural fit for streaming control

**⚠️ Borderline Cases:**

| Endpoint | Analysis | Recommendation |
|----------|----------|----------------|
| `POST /api/queue/clear` (client proxy) | Stateless operation, but proxies to server | **Keep** - provides useful abstraction |
| `POST /api/session/reset` | Stateless operation, but triggers server-side effect | **Keep** - atomic reset operation fits REST pattern |
| `POST /api/analysis/start` | Initializes stateful job, but returns immediately | **Keep** - initialization fits REST, execution via WebSocket |

#### WebSocket Endpoints (Should be stateful, streaming, bidirectional)

**✅ Good Examples:**

| Endpoint | Justification |
|----------|---------------|
| `WS /ws/stream` | High-bandwidth video frame streaming + inference results | Perfect fit: bidirectional, high-frequency data |
| `WS /ws/client` | Unified control plane: preview + commands + results | Perfect fit: persistent connection, multiple message types |
| `WS /ws/analysis` | Progress updates + result streaming during analysis | Perfect fit: progress streaming requires push |
| `WS /ws/logs` | Real-time server log broadcasting | Perfect fit: server-initiated push, no polling |

**No Violations Found:** All WebSocket endpoints are properly used for streaming/real-time data.

### Architecture Compliance Matrix

| Operation Type | Protocol | Examples | Compliance |
|---------------|----------|----------|------------|
| **Initialize resources** | HTTP POST | `/api/analysis/start`, `/api/session/init` | ✅ Mostly compliant (1 deprecated) |
| **Query data** | HTTP GET | `/api/datasets`, `/api/sessions`, `/api/status` | ✅ Fully compliant |
| **Update config** | HTTP POST/PUT | `/api/config`, `/api/camera/select` | ✅ Fully compliant |
| **Delete resources** | HTTP DELETE | `/api/sessions/{id}`, `/api/sessions/{id}/logs` | ✅ Fully compliant |
| **Start/stop streaming** | ~~HTTP~~ **WebSocket** | ~~`/api/start`~~ → `WS /ws/client {type: "start"}` | ⚠️ Transitioning |
| **Stream video frames** | WebSocket | `WS /ws/stream` | ✅ Fully compliant |
| **Real-time results** | WebSocket | `WS /ws/stream`, `WS /ws/client` | ✅ Fully compliant |
| **Progress updates** | WebSocket | `WS /ws/analysis` | ✅ Fully compliant |
| **Server logs** | WebSocket | `WS /ws/logs` | ✅ Fully compliant |

**Compliance Score:** 95% (3 deprecated endpoints remaining from transition)

### Evolution Timeline

| Phase | HTTP Usage | WebSocket Usage | Compliance |
|-------|-----------|-----------------|------------|
| **v0.x (Original)** | Commands + config + queries | Video frames only | 60% |
| **v1.x (Current)** | Config + queries + initialization | Commands + streaming + events | 95% |
| **v2.x (Target)** | Config + queries only | Commands + streaming + events | 100% |

**Target Architecture (v2.0):**
- **HTTP:** Pure stateless operations (config, queries, resource initialization)
- **WebSocket:** All stateful operations (streaming, commands tied to connection lifecycle)

---

## Part 5: Deprecation Analysis & Timeline

### High Confidence Deprecations (Remove in v2.0)

#### 1. Legacy Start/Stop Endpoints

**Endpoints:**
- `POST /api/start` ([routes.py:142](../src/iris/client/web/routes.py#L142))
- `POST /api/stop` ([routes.py:184](../src/iris/client/web/routes.py#L184))

**Superseded By:**
- `WS /ws/client` messages: `{type: "start", config: {...}}` and `{type: "stop"}`

**Rationale:**
- HTTP doesn't fit persistent streaming control
- WebSocket connection lifecycle naturally manages resource cleanup
- New architecture already implemented and working

**Migration Path:**

**Phase 1 (v1.x):** Add deprecation warnings
```python
@api_router.post("/start")
async def start_streaming(request: StartRequest | None = None) -> dict[str, Any]:
    logger.warning("DEPRECATED: /api/start endpoint. Use WS /ws/client with {type: 'start'} message")
    # Keep existing implementation
    ...
    return {
        "status": "ok",
        "message": "Streaming started",
        "warning": "This endpoint is deprecated. Use WebSocket /ws/client instead."
    }
```

**Phase 2 (v1.x+1):** Return HTTP 410 Gone
```python
@api_router.post("/start")
async def start_streaming(request: StartRequest | None = None) -> dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="This endpoint has been removed. Use WebSocket /ws/client with {type: 'start'} message."
    )
```

**Phase 3 (v2.0):** Remove endpoint entirely

**Frontend Changes Required:**
```typescript
// OLD (deprecated):
await fetch('/api/start', {method: 'POST', body: JSON.stringify({frames_per_segment: 8})})

// NEW:
websocket.send(JSON.stringify({type: 'start', config: {frames_per_segment: 8}}))
```

**Verification:**
```bash
# Check if frontend still calls these endpoints:
grep -r "/api/start" src/iris/client/web/frontend/
grep -r "/api/stop" src/iris/client/web/frontend/
```

**Timeline:**
- v1.2 (2 weeks): Add warnings
- v1.3 (1 month): Return 410 Gone
- v2.0 (3 months): Remove

---

#### 2. Session Init Endpoint

**Endpoint:**
- `POST /api/session/init` ([routes.py:49](../src/iris/client/web/routes.py#L49))

**Superseded By:**
- Auto-creation on `WS /ws/client` connect ([routes.py:1494-1501](../src/iris/client/web/routes.py#L1494-L1501))

**Rationale:**
- Sessions now auto-created when frontend connects to `/ws/client`
- `session_store` global dict is underutilized (only written, never read elsewhere)
- Adds unnecessary initialization step

**Evidence:**
```python
# routes.py:1494-1501 (WS /ws/client handler)
db_session = session_repo.get(state.session_id)
if not db_session:
    # Create new session in database
    session_repo.create(
        session_id=state.session_id,
        config=state.session_config,
    )
    logger.info(f"Created new session in DB: {state.session_id}")
```

**Migration Path:**

**Phase 1 (v1.x):** Verify `session_store` only used by `/api/session/init`
```bash
git grep "session_store" src/
# Expected: Only in routes.py lines 41-43 (definition) and 49-58 (init endpoint)
```

**Phase 2 (v2.0):** Remove both `session_store` and `/api/session/init` in same commit

**Frontend Changes Required:**
```typescript
// OLD (deprecated):
const response = await fetch('/api/session/init', {
  method: 'POST',
  body: JSON.stringify({frames_per_segment: 8, overlap_frames: 4})
})
const {session_id} = await response.json()

// NEW (automatic):
const ws = new WebSocket('/ws/client')
// Wait for session_info message:
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'session_info') {
    const session_id = data.session_id
    // Use session_id
  }
}
```

**Timeline:**
- v1.2: Verify usage
- v2.0: Remove

---

#### 3. Browser Camera Mode (Already Removed)

**Status:** Code removed, artifacts remain

**Artifacts:**
- Comment on [routes.py:38](../src/iris/client/web/routes.py#L38): `# camera_mode removed - server-only camera selection`
- `session_store` partially related to old camera mode workflow

**Cleanup:**
- Already complete on backend
- May have remnants in frontend (check for `camera_mode` references)

**Verification:**
```bash
grep -r "camera_mode" src/
# Frontend may still have CameraMode type definition
```

**Timeline:** Already complete (v1.x)

---

### Medium Confidence Deprecations (Consider for v2.0)

#### 1. Duplicate Health Endpoint in app.py

**Endpoint:**
- `GET /health` ([app.py:388](../src/iris/server/app.py#L388))

**Action:** Remove duplicate (see Redundancy Analysis § 2.1)

**Timeline:** v1.x patch (immediate)

---

#### 2. Client Queue Clear Proxy

**Endpoint:**
- `POST /api/queue/clear` ([routes.py:1437](../src/iris/client/web/routes.py#L1437))

**Recommendation:** Keep for now, reconsider in v3.0

**Rationale:** Low cost, useful abstraction (see Redundancy Analysis § 2.2)

**Timeline:** v3.0 (long-term)

---

### Low Confidence (Keep, Mark Experimental)

#### 1. Job Management Endpoints

**Endpoints:**
- All in [routes/jobs.py](../src/iris/server/routes/jobs.py)

**Status:** Dormant (not used by client backend)

**Recommendation:** Keep, mark as "Experimental" in API docs

**Rationale:**
- Enables future external API integrations
- REST-based job submission for scripting/automation
- Alternative to WebSocket for batch processing
- Minimal maintenance cost

**Documentation:**
```markdown
### Job Management API (Experimental)

**Status:** Experimental - Not currently used by official client

These endpoints provide a REST-based API for job submission and management.
Useful for:
- External integrations without WebSocket support
- Scripting and automation
- Batch processing workflows

**Note:** The official client uses WebSocket streaming (`/ws/stream`) instead.
```

**Timeline:** Keep indefinitely as experimental feature

---

### Deprecation Summary Table

| Endpoint | Status | Timeline | Action |
|----------|--------|----------|--------|
| `POST /api/start` | **Deprecated** | Remove v2.0 | Replace with WS message |
| `POST /api/stop` | **Deprecated** | Remove v2.0 | Replace with WS message |
| `POST /api/session/init` | **Deprecated** | Remove v2.0 | Auto-create on WS connect |
| `GET /health` (app.py) | **Duplicate** | Remove v1.x | Consolidate to router version |
| `POST /api/queue/clear` (client proxy) | **Borderline** | Keep v1-2, reconsider v3.0 | Low priority |
| Job management endpoints | **Experimental** | Keep indefinitely | Document as experimental |

---

## Part 6: Recommendations

### Immediate Actions (v1.x Patch)

#### 1. Remove Duplicate Health Endpoint **[HIGH PRIORITY]**

**Problem:** Two `/health` endpoints with different responses

**Action:**
1. Enhance router version ([routes/system.py:14](../src/iris/server/routes/system.py#L14)) to include `queue_depth`
2. Remove inline version ([app.py:388](../src/iris/server/app.py#L388))
3. Update all callers to expect consistent response format

**Code Changes:**
```python
# src/iris/server/routes/system.py:14
@router.get("/health")
async def health() -> dict[str, str | bool | int]:
    state = get_server_state()
    queue_depth = state.queue.queue.qsize() if state.queue and state.queue.queue else 0
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
        "queue_depth": queue_depth,  # ADD THIS FIELD
    }
```

```python
# src/iris/server/app.py:388
# DELETE LINES 388-401 (entire health_check function)
```

**Testing:**
```bash
# Verify health endpoint works via main.py entry point:
python -m iris.server.main &
curl http://localhost:8001/health
# Expected: {"status": "healthy", "model_loaded": true, "queue_depth": 0}
```

**Impact:** Low risk, high value (reduces confusion)

**Timeline:** Next patch release (1 week)

---

#### 2. Add Deprecation Warnings to Legacy Endpoints **[MEDIUM PRIORITY]**

**Endpoints:**
- `POST /api/start`
- `POST /api/stop`
- `POST /api/session/init`

**Action:** Add log warnings and response warnings

**Code Changes:**
```python
@api_router.post("/start")
async def start_streaming(request: StartRequest | None = None) -> dict[str, Any]:
    logger.warning(
        "DEPRECATED: /api/start endpoint called. "
        "Use WebSocket /ws/client with {type: 'start'} message instead. "
        "This endpoint will be removed in v2.0."
    )
    # Keep existing implementation
    ...
    return {
        "status": "ok",
        "message": "Streaming started",
        "deprecation_warning": {
            "message": "This endpoint is deprecated and will be removed in v2.0",
            "migration": "Use WebSocket /ws/client with {type: 'start'} message",
            "docs": "https://iris-docs.example.com/migration-guide"
        }
    }
```

**Frontend Impact:** Frontend should display deprecation warning in console

**Timeline:** Next minor release (2 weeks)

---

### Short-Term Actions (v2.0)

#### 1. Consolidate Server Entry Points **[HIGH PRIORITY]**

**Problem:** `app.py` and `main.py` both serve as entry points with overlapping functionality

**Action:** Migrate all routes from `app.py` to modular routers, remove `app.py`

**Migration Plan:**

**Step 1:** Create new routers

```python
# src/iris/server/routes/config.py (NEW)
from fastapi import APIRouter
router = APIRouter()

@router.get("/api/config/defaults")
async def get_config_defaults() -> dict:
    # Move from app.py:304-343
    ...
```

```python
# src/iris/server/routes/websocket.py (NEW)
from fastapi import APIRouter, WebSocket
router = APIRouter()

@router.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    # Move from app.py:404-986
    ...

@router.websocket("/ws/logs")
async def log_streaming_endpoint(websocket: WebSocket) -> None:
    # Move from app.py:988-1020
    ...
```

**Step 2:** Consolidate queue clear

```python
# src/iris/server/routes/system.py
# ADD:
@router.post("/api/queue/clear")
async def clear_queue() -> dict:
    # Move from app.py:346-385
    # Or consolidate with /system/clear
    ...
```

**Step 3:** Update main.py

```python
# src/iris/server/main.py
from iris.server.routes import jobs, system, config, websocket

app.include_router(jobs.router)
app.include_router(system.router)
app.include_router(config.router)  # NEW
app.include_router(websocket.router)  # NEW
```

**Step 4:** Update startup scripts, documentation

**Step 5:** Remove `app.py`

**Benefits:**
- Single source of truth
- Better code organization
- Easier testing (individual router tests)
- Clearer architectural boundaries

**Timeline:** v2.0 (3 months)

---

#### 2. Remove Deprecated HTTP Endpoints **[HIGH PRIORITY]**

**Endpoints:**
- `POST /api/start`
- `POST /api/stop`
- `POST /api/session/init`

**Prerequisites:**
- Frontend migration complete (verified via grep)
- Deprecation warnings in place for 2+ releases

**Action:** Delete endpoint handlers

**Testing:** E2E tests should verify WebSocket-based workflow works

**Timeline:** v2.0 (3 months)

---

#### 3. Enhanced Queue Clear Validation **[MEDIUM PRIORITY]**

**Problem:** Queue clear failure during session reset is non-fatal, may lead to ghost results

**Action:** Add validation that queue is actually empty after clear

**Code Changes:**
```python
# src/iris/client/web/routes.py:1381 (session reset)
async def reset_session() -> dict[str, Any]:
    # ... existing reset logic ...

    # ENHANCED: Clear queue with validation
    try:
        response = await client.post(f"http://{host}:{port}/api/queue/clear")
        if response.status == 200:
            data = await response.json()
            cleared = data.get("cleared", 0)
            logger.info(f"Cleared {cleared} jobs from inference queue")

            # VALIDATION: Verify queue is empty
            health_response = await client.get(f"http://{host}:{port}/health")
            if health_response.status == 200:
                health_data = await health_response.json()
                queue_depth = health_data.get("queue_depth", -1)
                if queue_depth > 0:
                    logger.warning(
                        f"Queue not empty after clear: {queue_depth} jobs remaining. "
                        "May experience ghost results."
                    )
                else:
                    logger.info("Queue verified empty after clear")
    except Exception as e:
        logger.error(f"Failed to clear/verify inference queue: {e}")
        # Still allow reset to proceed (non-fatal)

    # ... rest of reset logic ...
```

**Timeline:** v2.0

---

### Long-Term Actions (v3.0)

#### 1. Direct Server Communication **[LOW PRIORITY]**

**Current:** Frontend → Client Backend → Inference Server

**Proposed:** Frontend → Inference Server (direct)

**Action:** Remove client backend proxy for queue clear

**Rationale:**
- Frontend already has server URL in config
- Removes indirection layer
- Makes data flow more explicit

**Trade-offs:**
- Breaks abstraction (frontend now knows about server topology)
- Requires frontend to manage two connections (client backend + inference server)

**Recommendation:** Only if frontend becomes more sophisticated (e.g., direct `/ws/logs` connection for observability)

**Timeline:** v3.0+ (12+ months)

---

#### 2. Session Management Review **[LOW PRIORITY]**

**Current:** Manager-Worker pattern (client owns session_id, server is stateless)

**Alternative:** Enhanced queue isolation (per-session queues on server)

**Analysis:** Current architecture is sound (per CLAUDE.md design), but could explore:
- Per-session queue isolation (prevent cross-session interference)
- Server-side session expiration (cleanup stale sessions)
- Distributed session management (multi-server deployments)

**Recommendation:** Review after real-world usage data collected

**Timeline:** v3.0+ (12+ months)

---

### Recommendations Summary Table

| Recommendation | Priority | Timeline | Impact |
|---------------|----------|----------|--------|
| Remove duplicate `/health` | High | v1.x (1 week) | Low risk, high value |
| Add deprecation warnings | Medium | v1.x+1 (2 weeks) | No breaking changes |
| Consolidate server entry points | High | v2.0 (3 months) | Refactoring effort |
| Remove deprecated endpoints | High | v2.0 (3 months) | Breaking change |
| Enhanced queue validation | Medium | v2.0 (3 months) | Improved robustness |
| Direct server communication | Low | v3.0+ (12+ months) | Architectural change |
| Session management review | Low | v3.0+ (12+ months) | Design exploration |

---

## Part 7: Testing Gaps

### Integration Tests Needed

#### 1. Session Reset Race Condition Test **[HIGH PRIORITY]**

**Scenario:** Verify queue clear prevents ghost results after session reset

**Test Case:**
```python
async def test_session_reset_prevents_ghost_results():
    """
    Verify that resetting session clears inference queue,
    preventing old results from appearing in new session.
    """
    # Setup: Start inference server and client backend
    inference_server = await start_inference_server()
    client_backend = await start_client_backend()

    # Step 1: Start streaming and enqueue 10 inference jobs
    ws = await connect_websocket("/ws/client")
    await ws.send_json({"type": "start", "config": {...}})

    # Send 10 segments worth of frames
    for i in range(10):
        frames = create_test_segment(i)
        await send_frames_to_server(frames)

    # Verify queue has jobs
    health = await client.get(f"http://{inference_server.url}/health")
    assert health.json()["queue_depth"] >= 5, "Queue should have pending jobs"

    # Step 2: Immediately reset session
    old_session_id = ws.received_session_id
    response = await client.post("/api/session/reset")
    new_session_id = response.json()["session_id"]
    assert new_session_id != old_session_id

    # Step 3: Verify queue was cleared
    health = await client.get(f"http://{inference_server.url}/health")
    assert health.json()["queue_depth"] == 0, "Queue should be empty after reset"

    # Step 4: Wait for any in-flight results
    await asyncio.sleep(5)

    # Step 5: Verify no results associated with new session
    results = await client.get(f"/api/sessions/{new_session_id}/results")
    assert len(results.json()["results"]) == 0, "New session should have no results"

    # Step 6: Verify old session results not polluting new session
    # (This would be a ghost result bug)
```

**Priority:** High (prevents data integrity bug)

**Estimated Effort:** 1 day

---

#### 2. Analysis Handshake Protocol Test **[HIGH PRIORITY]**

**Scenario:** Verify `complete` → `processing_complete` handshake ensures all results delivered

**Test Case:**
```python
async def test_analysis_completion_handshake():
    """
    Verify that analysis handshake protocol ensures all frames
    are processed before connection closes.
    """
    # Setup: Prepare test video (100 frames, 10 segments)
    video_path = create_test_video(frames=100, fps=10)

    # Step 1: Start analysis
    response = await client.post("/api/analysis/start", json={
        "video_filename": video_path.name,
        "segment_time": 1.0,
        "frames_per_segment": 10,
        "overlap_frames": 0,
    })
    job_id = response.json()["job_id"]
    expected_segments = 10

    # Step 2: Connect to analysis WebSocket
    ws = await connect_websocket("/ws/analysis")

    # Step 3: Collect all results
    results = []
    upload_complete_received = False
    processing_complete_received = False

    async for message in ws:
        data = json.loads(message)

        if data["type"] == "result":
            results.append(data)
        elif data["type"] == "upload_complete":
            upload_complete_received = True
            logger.info("Upload complete, waiting for processing_complete")
        elif data["type"] == "complete":
            processing_complete_received = True
            logger.info("Processing complete signal received")
            break

    # Step 4: Verify all signals received
    assert upload_complete_received, "Should receive upload_complete"
    assert processing_complete_received, "Should receive processing_complete"

    # Step 5: Verify all segments processed
    assert len(results) == expected_segments, (
        f"Expected {expected_segments} results, got {len(results)}. "
        "Handshake may have failed, losing results."
    )

    # Step 6: Verify results stored in database
    db_results = await client.get(f"/api/sessions/{job_id}/results")
    assert len(db_results.json()["results"]) == expected_segments
```

**Priority:** High (prevents data loss)

**Estimated Effort:** 1 day

---

#### 3. Duplicate Endpoints Compatibility Test **[MEDIUM PRIORITY]**

**Scenario:** Verify duplicate `/health` endpoints return compatible responses

**Test Case:**
```python
async def test_health_endpoints_compatibility():
    """
    Verify that duplicate health endpoints return compatible responses.
    """
    # Health endpoint via app.py (inline route)
    app_health = await client.get("http://localhost:8001/health")
    app_data = app_health.json()

    # Health endpoint via main.py (router)
    main_health = await client.get("http://localhost:8001/health")
    main_data = main_health.json()

    # Verify both have required fields
    assert "status" in app_data
    assert "model_loaded" in app_data
    assert "status" in main_data
    assert "model_loaded" in main_data

    # After fix, verify both have queue_depth
    assert "queue_depth" in app_data
    assert "queue_depth" in main_data

    # Verify compatible values
    assert app_data["model_loaded"] == main_data["model_loaded"]
    assert app_data["queue_depth"] == main_data["queue_depth"]
```

**Priority:** Medium (validates fix for redundancy)

**Estimated Effort:** 2 hours

---

#### 4. WebSocket Reconnection Test **[MEDIUM PRIORITY]**

**Scenario:** Verify WebSocket reconnection preserves session state

**Test Case:**
```python
async def test_websocket_reconnection_preserves_state():
    """
    Verify that reconnecting to /ws/client preserves session state.
    """
    # Step 1: Connect, get session_id
    ws1 = await connect_websocket("/ws/client")
    session_info = await ws1.receive_json()
    assert session_info["type"] == "session_info"
    session_id1 = session_info["session_id"]

    # Step 2: Start inference, receive some results
    await ws1.send_json({"type": "start", "config": {...}})
    await asyncio.sleep(5)  # Let some inference happen

    # Step 3: Disconnect
    await ws1.close()

    # Step 4: Reconnect
    ws2 = await connect_websocket("/ws/client")
    session_info2 = await ws2.receive_json()
    session_id2 = session_info2["session_id"]

    # Step 5: Verify session restored (or new session created)
    # Current behavior: New session created
    # Future behavior: Should restore from DB if session_id in cookie

    # Step 6: Verify results from first session accessible
    results = await client.get(f"/api/sessions/{session_id1}/results")
    assert len(results.json()["results"]) > 0, "First session results should be preserved"
```

**Priority:** Medium (validates session restoration)

**Estimated Effort:** 4 hours

---

### End-to-End Tests Needed

#### 1. Full Analysis Workflow **[HIGH PRIORITY]**

**Scenario:** Complete analysis from video upload to report generation

**Test Case:**
```python
async def test_full_analysis_workflow():
    """End-to-end test of complete analysis workflow."""
    # 1. Upload test video
    video_path = create_test_video(duration_sec=10, fps=10)

    # 2. List datasets, verify video appears
    datasets = await client.get("/api/datasets")
    assert video_path.name in [v["filename"] for v in datasets.json()["videos"]]

    # 3. Start analysis
    response = await client.post("/api/analysis/start", json={
        "video_filename": video_path.name,
        "segment_time": 1.0,
        "frames_per_segment": 10,
        "overlap_frames": 0,
    })
    job_id = response.json()["job_id"]

    # 4. Connect to analysis WebSocket, monitor progress
    ws = await connect_websocket("/ws/analysis")

    results_count = 0
    async for message in ws:
        data = json.loads(message)
        if data["type"] == "result":
            results_count += 1
        elif data["type"] == "complete":
            break

    # 5. Verify all results delivered
    assert results_count == 10, "Expected 10 results for 10-second video"

    # 6. Verify results persisted in database
    db_results = await client.get(f"/api/sessions/{job_id}/results")
    assert len(db_results.json()["results"]) == 10

    # 7. Generate report
    report_response = await client.post("/api/report/generate", json={
        "session_id": job_id
    })
    report = await report_response.text()
    assert len(report) > 100, "Report should contain analysis"

    # 8. Verify report stored
    stored_report = await client.get(f"/api/report/{job_id}")
    assert stored_report.json()["report"]["content"] == report

    # 9. Cleanup
    await client.delete(f"/api/sessions/{job_id}")
```

**Priority:** High (validates entire analysis flow)

**Estimated Effort:** 1 day

---

#### 2. Live Streaming with Interruption **[MEDIUM PRIORITY]**

**Scenario:** Start live streaming, disconnect, verify cleanup

**Test Case:**
```python
async def test_live_streaming_with_interruption():
    """Test live streaming with unexpected disconnect."""
    # 1. Connect to /ws/client
    ws = await connect_websocket("/ws/client")
    session_info = await ws.receive_json()
    session_id = session_info["session_id"]

    # 2. Start inference
    await ws.send_json({"type": "start", "config": {...}})

    # 3. Receive preview frames and results
    frames_received = 0
    results_received = 0

    for _ in range(10):
        message = await ws.receive_json()
        if message["type"] == "preview_frame":
            frames_received += 1
        elif message["type"] == "result":
            results_received += 1

    assert frames_received > 0, "Should receive preview frames"

    # 4. Simulate unexpected disconnect (no stop message)
    await ws.close()

    # 5. Wait for server cleanup
    await asyncio.sleep(2)

    # 6. Verify server cleaned up resources
    # (Check server logs for cleanup messages)

    # 7. Reconnect
    ws2 = await connect_websocket("/ws/client")
    session_info2 = await ws2.receive_json()

    # 8. Verify can start streaming again (no resource leak)
    await ws2.send_json({"type": "start", "config": {...}})
    message = await ws2.receive_json()
    # Should not get error about camera already in use
```

**Priority:** Medium (validates resource cleanup)

**Estimated Effort:** 4 hours

---

### Testing Summary

| Test Category | Priority | Count | Total Effort |
|--------------|----------|-------|--------------|
| Integration Tests | High | 2 | 2 days |
| Integration Tests | Medium | 2 | 6 hours |
| E2E Tests | High | 1 | 1 day |
| E2E Tests | Medium | 1 | 4 hours |
| **Total** | | **6 tests** | **~4 days** |

**Test Infrastructure Needed:**
- Async test fixtures (pytest-asyncio)
- WebSocket test client (websockets library)
- Mock inference server (for isolated client tests)
- Test video generation utilities
- Database fixtures (SQLite in-memory)

**Test Execution:**
```bash
# Run integration tests:
pytest tests/integration/ -v

# Run E2E tests:
pytest tests/e2e/ -v --slow

# Run all tests:
pytest tests/ -v
```

---

## Appendix

### A. Complete Endpoint Inventory (CSV)

```csv
Component,Type,Path,Method,File,Line,Status,Notes
Server,HTTP,/api/config/defaults,GET,app.py,304,Active,Returns config defaults
Server,HTTP,/api/queue/clear,POST,app.py,346,Active,Clear queue + GC
Server,HTTP,/health,GET,app.py,388,DUPLICATE,Remove (use router version)
Server Router,HTTP,/health,GET,routes/system.py,14,Active,Preferred version
Server Router,HTTP,/metrics,GET,routes/system.py,24,Active,Detailed metrics
Server Router,HTTP,/system/clear,DELETE,routes/system.py,42,Active,Full system reset
Server Router,HTTP,/jobs/start,POST,routes/jobs.py,13,Dormant,Job-based API
Server Router,HTTP,/jobs/{id}/stop,DELETE,routes/jobs.py,44,Dormant,Stop job
Server Router,HTTP,/jobs/{id}/status,GET,routes/jobs.py,75,Dormant,Job status
Server Router,HTTP,/jobs/{id}/trigger,POST,routes/jobs.py,94,Dormant,Manual trigger
Server Router,HTTP,/jobs/active,GET,routes/jobs.py,125,Dormant,List jobs
Server,WebSocket,/ws/stream,WS,app.py,404,Active,Primary data plane
Server,WebSocket,/ws/logs,WS,app.py,988,Active,Log streaming
Client,HTTP,/api/status,GET,routes.py,68,Active,Client status
Client,HTTP,/api/config,POST,routes.py,80,Active,Update config
Client,HTTP,/api/config/defaults,GET,routes.py,127,Active,Get defaults
Client,HTTP,/api/config/gemini-key,GET,routes.py,97,Active,Check API key
Client,HTTP,/api/config/gemini-key,POST,routes.py,113,Active,Store API key
Client,HTTP,/api/session/init,POST,routes.py,49,DEPRECATED,Remove in v2.0
Client,HTTP,/api/session/reset,POST,routes.py,1381,Active,Critical for reset
Client,HTTP,/api/cameras,GET,routes.py,204,Active,List cameras
Client,HTTP,/api/camera/select,POST,routes.py,228,Active,Switch camera
Client,HTTP,/api/start,POST,routes.py,142,DEPRECATED,Remove in v2.0
Client,HTTP,/api/stop,POST,routes.py,184,DEPRECATED,Remove in v2.0
Client,HTTP,/api/results/history,GET,routes.py,255,Active,Get results
Client,HTTP,/api/results/clear,POST,routes.py,265,Active,Clear results
Client,HTTP,/api/datasets,GET,routes.py,278,Active,List videos
Client,HTTP,/api/videos/{filename},GET,routes.py,327,Active,Serve video
Client,HTTP,/api/analysis/start,POST,routes.py,436,Active,Start analysis
Client,HTTP,/api/analysis/stop,POST,routes.py,657,Active,Stop analysis
Client,HTTP,/api/sessions,GET,routes.py,1076,Active,List sessions
Client,HTTP,/api/sessions/{id},GET,routes.py,1085,Active,Get session
Client,HTTP,/api/sessions/{id},DELETE,routes.py,1132,Active,Delete session
Client,HTTP,/api/sessions/{id}/results,GET,routes.py,1096,Active,Get results
Client,HTTP,/api/sessions/{id}/results,DELETE,routes.py,1123,Active,Clear results
Client,HTTP,/api/sessions/{id}/logs,GET,routes.py,1105,Active,Get logs
Client,HTTP,/api/sessions/{id}/logs,DELETE,routes.py,1114,Active,Clear logs
Client,HTTP,/api/session/{id}/data,GET,routes.py,1342,Active,Get session data
Client,HTTP,/api/report/generate,POST,routes.py,1146,Active,Generate report
Client,HTTP,/api/report/{id},GET,routes.py,1323,Active,Get stored report
Client,HTTP,/api/report/fallback/{id},GET,routes.py,1289,Active,Fallback report
Client,HTTP,/api/queue/clear,POST,routes.py,1437,Active,Proxy to server
Client,WebSocket,/ws/client,WS,routes.py,1461,Active,Unified control plane
Client,WebSocket,/ws/analysis,WS,routes.py,679,Active,Analysis progress
```

**Total Endpoints:** 44
- **Server:** 13 (3 HTTP inline, 8 HTTP router, 2 WebSocket)
- **Client:** 31 (29 HTTP, 2 WebSocket)

---

### B. WebSocket Message Type Catalog

#### /ws/stream (Inference Server)

**Client → Server:**
- `session_config` - Initialize session with segment parameters
- `frame` - Send video frame for inference
- `complete` - Signal end of frame stream (analysis mode only)

**Server → Client:**
- `session_ack` - Confirm session creation, return session_id
- `result` - Inference result for completed segment
- `session_metrics` - Real-time metrics (fps, queue_depth, total_frames)
- `log` - Server log message
- `processing_complete` - All batches processed (response to `complete`)
- `error` - Error message

#### /ws/client (Client Backend)

**Client → Server:**
- `start` - Start camera and inference with config
- `stop` - Stop inference (keep camera running)
- `clear_queue` - Clear inference server queue
- `reset_session` - Reset session and generate new session_id

**Server → Client:**
- `session_info` - Session ID and config (sent on connect)
- `preview_frame` - USB camera preview frame (10 FPS)
- `result` - Inference result (forwarded from inference server)
- `metrics` - Session metrics (forwarded from inference server)
- `server_status` - Inference server health status (every 5 seconds)
- `error` - Error message
- `log` - Log message

#### /ws/analysis (Client Backend)

**Client → Server:**
- (None - receive only)

**Server → Client:**
- `session_ack` - Session acknowledged (forwarded from inference server)
- `session_metrics` - Real-time metrics (forwarded from inference server)
- `result` - Inference result (augmented with frame_range, video_time_ms)
- `progress` - Upload progress (frames_sent, total_frames, eta_sec, fps) [max 10/sec]
- `upload_complete` - All frames sent to inference server
- `complete` - Analysis fully complete (all processing done)
- `error` - Error message
- `log` - Log message

#### /ws/logs (Inference Server)

**Client → Server:**
- (Keep-alive pings)

**Server → Client:**
- `log` - Python logging record `{type: "log", level, name, message, timestamp}`
- `ping` - Keep-alive (every 30 seconds)

---

### C. File Reference Map

**Server Files:**
- [src/iris/server/app.py](../src/iris/server/app.py) - Legacy monolithic entry point
- [src/iris/server/main.py](../src/iris/server/main.py) - Modern modular entry point
- [src/iris/server/routes/jobs.py](../src/iris/server/routes/jobs.py) - Job management router
- [src/iris/server/routes/system.py](../src/iris/server/routes/system.py) - System management router
- [src/iris/server/config.py](../src/iris/server/config.py) - Server configuration
- [src/iris/server/dependencies.py](../src/iris/server/dependencies.py) - Dependency injection
- [src/iris/server/lifecycle.py](../src/iris/server/lifecycle.py) - Lifecycle handler
- [src/iris/server/inference/executor.py](../src/iris/server/inference/executor.py) - Inference executor
- [src/iris/server/jobs/manager.py](../src/iris/server/jobs/manager.py) - Job manager

**Client Files:**
- [src/iris/client/web/routes.py](../src/iris/client/web/routes.py) - All client endpoints
- [src/iris/client/web/app.py](../src/iris/client/web/app.py) - Client app initialization
- [src/iris/client/web/dependencies.py](../src/iris/client/web/dependencies.py) - App state
- [src/iris/client/web/messages.py](../src/iris/client/web/messages.py) - Message type definitions
- [src/iris/client/web/repositories.py](../src/iris/client/web/repositories.py) - Database repos
- [src/iris/client/web/report_generator.py](../src/iris/client/web/report_generator.py) - Report generation
- [src/iris/client/streaming/websocket_client.py](../src/iris/client/streaming/websocket_client.py) - StreamingClient
- [src/iris/client/capture/camera.py](../src/iris/client/capture/camera.py) - CameraCapture
- [src/iris/client/capture/video_file.py](../src/iris/client/capture/video_file.py) - VideoFileCapture

---

### D. References

- **IRIS Project Guide:** [CLAUDE.md](../CLAUDE.md)
- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **WebSocket Protocol:** RFC 6455
- **OpenAPI 3.1 Spec:** https://spec.openapis.org/oas/v3.1.0

---

## Conclusion

This document provides a complete audit of all HTTP and WebSocket endpoints in the IRIS system. Key takeaways:

1. **3 Redundancies Identified:** Duplicate health endpoints, queue clear proxy, dual entry points
2. **4 Deprecations Confirmed:** Browser camera mode, legacy start/stop, session init, dormant job endpoints
3. **Architecture Evolution:** Successfully transitioned from HTTP-first to WebSocket-first (95% compliant)
4. **Recommendations Prioritized:** Immediate fixes (duplicate health), short-term cleanup (v2.0 deprecation removal), long-term exploration (v3.0 architecture review)
5. **Testing Gaps Identified:** 6 critical tests needed (session reset race condition, analysis handshake, etc.)

The codebase is **currently functioning** and this analysis provides a roadmap for **"ironing out the details"** to achieve clean, maintainable, and well-documented API architecture.
