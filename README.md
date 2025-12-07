# IRIS

A research project done with the AI Team by Myriam Benlamri (Data Science MSc 2nd year) and Marcus Hamelink (Computer Science BSc 3rd year) as a collaborative research semester project.

More info at [https://epflaiteam.ch/projects/iris](https://epflaiteam.ch/projects/iris)


## Set up

### Client

On Your Raspberry Pi, generate the self-signed certificate:
```bash
mkdir -p ~/iris-certs
cd ~/iris-certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=$(hostname -I | awk '{print $1}')"
```

To use HTTPS, make sure to specify `use_ssl=true` under `client` in `config.yaml`. In that case, make sure to connect to the HTTPS IP.

Run `uv run iris-client` to start a client instance.


### Server

Run `uv run iris-server` to start a server instance.


## Job System

IRIS uses a flexible job system for managing inference tasks. Jobs can be started via API and triggered in multiple ways.

### Job Types

**1. SingleFrameJob**
- Processes each incoming frame individually with VLM
- Useful for: real-time inference, testing, continuous monitoring
- Trigger: Automatic on every frame (or every Nth frame with `frame_skip`)

**2. VideoJob**
- Collects frames in buffer, then processes batch with video-aware VLM
- Useful for: temporal understanding, action recognition, video summarization
- Supports three trigger modes: periodic (automatic), manual (API), and disabled (job-to-job)

### Trigger Modes

VideoJob supports three triggering modes via the `trigger_mode` parameter:

**PERIODIC (Automatic)**
- Automatically triggers inference when buffer reaches `buffer_size` frames
- After inference, keeps last `overlap_frames` for temporal continuity
- Configuration example:
```python
{
    "job_type": "video",
    "trigger_mode": "periodic",
    "buffer_size": 8,
    "overlap_frames": 4
}
```
**Use case:** Continuous video analysis (e.g., Qwen2.5-VL logging)

**MANUAL (API-triggered)**
- Buffers frames but only triggers via API call: `POST /jobs/{job_id}/trigger`
- No overlap - buffer clears after each trigger
- Configuration example:
```python
{
    "job_type": "video",
    "trigger_mode": "manual",
    "buffer_size": 1
}
```
**Use case:** On-demand analysis (e.g., colony counter when user clicks)

**DISABLED (Buffering Only)**
- Accepts and buffers frames but never processes them
- For future use or conditional triggering
- Configuration example:
```python
{
    "job_type": "video",
    "trigger_mode": "disabled",
    "buffer_size": 8
}
```
**Use case:** Placeholder for future YOLO integration

### Auto-Started VideoJob

When a client connects to `/ws/stream`, a VideoJob is automatically created for that connection:
- Job ID: Unique per connection (e.g., `video_job_a3f7b2c1`)
- Mode: PERIODIC
- Buffer: 8 frames (configurable in `config.yaml`)
- Overlap: 4 frames - 50% overlap for temporal continuity
- Cleanup: Automatically stopped and removed when WebSocket disconnects

**No manual job creation needed - just start streaming!**

You can configure defaults in `config.yaml`:
```yaml
jobs:
  video:
    trigger_mode: "periodic"
    buffer_size: 8
    overlap_frames: 4
```

### API Endpoints

**Start a job:**
```bash
POST /jobs/start
Content-Type: application/json

{
    "job_type": "video",
    "prompt": "Describe what you see in the video.",
    "trigger_mode": "periodic",
    "buffer_size": 8,
    "overlap_frames": 4
}
```

**Manually trigger inference:**
```bash
POST /jobs/{job_id}/trigger
```

**Get job status:**
```bash
GET /jobs/{job_id}/status
```

**List active jobs:**
```bash
GET /jobs/active
```

**Stop a job:**
```bash
POST /jobs/{job_id}/stop
```

### WebSocket Logging

Jobs send progress logs via WebSocket (`/ws/stream`):

```json
{
    "type": "log",
    "job_id": "video-abc123",
    "message": "Buffered frame 3/5",
    "timestamp": 1234567890.123
}
```

Results are also sent via WebSocket:

```json
{
    "type": "result",
    "job_id": "video-abc123",
    "job_type": "video",
    "status": "completed",
    "result": "..."
}
```

### Job Orchestration

Jobs can launch other jobs during execution, enabling conditional workflows:

```python
class YOLOVideoJob(VideoJob):
    async def _run_inference(self):
        # Run YOLO detection
        detections = await self._run_yolo(self.frame_buffer)

        # If object detected, launch VLM job
        if detections["confidence"] > 0.5:
            vlm_config = VideoJobConfig(
                prompt="Describe what the detected object is doing.",
                trigger=TriggerConfig(mode=TriggerMode.DISABLED)
            )
            vlm_job = self.job_factory.create_job(vlm_config, ...)
            await self.queue.submit(vlm_job)
```

### Multi-GPU Support

Set `server.num_workers` in `config.yaml` to utilize multiple GPUs:

```yaml
server:
  num_workers: 2  # Uses 2 GPUs in round-robin
```

Workers are automatically assigned to GPUs: `worker_id % device_count`.

### Video Inference Notes

**TODO:** The current VideoJob implementation processes only the first frame as a placeholder. Proper video inference requires exploring Qwen2.5-VL's video prompt template, which may support native video input with special tokens for temporal understanding.

See `src/iris/vlm/inference/queue/jobs.py:VideoJob._sync_inference()` for implementation details.

## Workflow with Izar

This supposee

### On Izar
```
cd /path/to/IRIS
Sinteract -t 00:20:00 -g gpu:1 -m 32G -q team-ai
hostname
./run_iris.sh
```

### On Personal machine

**Terminal 1**
```
uv run iris-client
```

**Terminal 2**
```
ssh -N -L 8005:[RUN hostname ON NODE TO SEE]:8001 EPFL-USERNAME@izar.hpc.epfl.ch
```

Then go to http://localhost:8006

Important, make sure to modify the hostname
