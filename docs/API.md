# IRIS API Reference

## Server API (`iris-server`)

Default port: `8001`

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | System metrics and job stats |
| `POST` | `/jobs/start` | Start inference job |
| `DELETE` | `/jobs/{job_id}/stop` | Stop running job |
| `GET` | `/jobs/{job_id}/status` | Get job status |
| `POST` | `/jobs/{job_id}/trigger` | Manual inference trigger |
| `GET` | `/jobs/active` | List active jobs |
| `DELETE` | `/system/clear` | Reset server state |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/stream` | Frame streaming and inference results |
| `/ws/logs` | Real-time log streaming |

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `POST /jobs/start`

Start a new inference job.

**Request Body:**
```json
{
  "job_type": "video",
  "prompt": "Describe what you see",
  "buffer_size": 8,
  "overlap_frames": 4,
  "trigger_mode": "periodic",
  "default_fps": 5.0,
  "max_new_tokens": 128
}
```

**Response:**
```json
{
  "job_id": "video_job_abc123",
  "status": "started",
  "job_type": "video",
  "config": { ... }
}
```

---

### `DELETE /jobs/{job_id}/stop`

Stop a running job.

**Response:**
```json
{
  "job_id": "video_job_abc123",
  "status": "stopped",
  "message": "Job stopped successfully"
}
```

---

### `GET /metrics`

Get system metrics and recent job statistics.

**Response:**
```json
{
  "enable_metrics": true,
  "stats": {
    "total_jobs": 150,
    "completed_jobs": 145,
    "failed_jobs": 5,
    "dropped_frames": 0,
    "avg_inference_time": 125.5,
    "p50_latency": 120.0,
    "p95_latency": 180.0,
    "p99_latency": 220.0
  },
  "recent_jobs": [ ... ]
}
```

---

### `WS /ws/stream`

WebSocket endpoint for frame streaming and inference results.

**Frame Message (Client -> Server):**
```json
{
  "frame_id": 1,
  "timestamp": 1704067200.123,
  "fps": 10.0,
  "frame": "<base64-encoded-jpeg>"
}
```

**Result Message (Server -> Client):**
```json
{
  "job_id": "video_job_abc123_batch_0",
  "text": "A person is pipetting liquid into a vial",
  "inference_time": 125.5,
  "frame_count": 8,
  "client_fps": 10.0
}
```

---

## Client API (`iris-client`)

Default port: `8000`

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Client status |
| `POST` | `/api/start` | Start camera + streaming |
| `POST` | `/api/stop` | Stop streaming |
| `POST` | `/api/config` | Update server config |
| `GET` | `/api/cameras` | List available cameras |
| `GET` | `/api/datasets` | List video/annotation files |
| `POST` | `/api/analysis/start` | Start video analysis |
| `POST` | `/api/analysis/stop` | Stop analysis |
| `GET` | `/api/sessions` | List analysis sessions |
| `GET` | `/api/sessions/{id}` | Get session details |
| `GET` | `/api/sessions/{id}/results` | Get session results |
| `GET` | `/api/sessions/{id}/logs` | Get session logs |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `GET` | `/api/report/generate` | Stream LLM report |
| `GET` | `/api/report/fallback/{id}` | Basic stats report |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/preview` | Camera preview stream |
| `/ws/results` | Live inference results |
| `/ws/analysis` | Analysis progress updates |

---

### `GET /api/status`

Get current client status.

**Response:**
```json
{
  "camera_active": true,
  "streaming_active": true,
  "streaming_server_status": "connected",
  "fps": 10.5,
  "config": { ... }
}
```

---

### `POST /api/analysis/start`

Start video file analysis.

**Request Body:**
```json
{
  "video_filename": "experiment_001.mp4",
  "annotation_filename": "experiment_001.jsonl",
  "simulation_fps": 5.0
}
```

**Response:**
```json
{
  "status": "ok",
  "job_id": "analysis_abc123",
  "total_frames": 1500,
  "annotation_count": 45
}
```

---

### `GET /api/report/generate`

Stream an LLM-generated analysis report.

**Query Parameters:**
- `session_id` (required): Session ID
- `provider` (optional): `anthropic`, `openai`, or `gemini` (default: `anthropic`)

**Response:** Streaming text/plain with markdown content.

---

### `GET /api/report/fallback/{session_id}`

Get a basic statistics report without LLM.

**Response:**
```json
{
  "session_id": "analysis_abc123",
  "report": "# Analysis Report\n\n## Session Summary\n..."
}
```

---

## Environment Variables

### Server
- `IRIS_SERVER_HOST`: Server bind host (default: `0.0.0.0`)
- `IRIS_SERVER_PORT`: Server port (default: `8001`)

### Client (Report Generation)
- `ANTHROPIC_API_KEY`: For Claude reports
- `OPENAI_API_KEY`: For GPT reports
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`: For Gemini reports
