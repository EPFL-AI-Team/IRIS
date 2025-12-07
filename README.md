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

VideoJob supports three different triggering mechanisms:

**PERIODIC (Automatic)**
- Triggers when `frame_count >= N` OR `time_seconds >= T`
- Configuration example:
```python
{
    "job_type": "video",
    "trigger": {
        "mode": "periodic",
        "frame_count": 5,
        "time_seconds": 5.0
    }
}
```

**MANUAL (API-triggered)**
- Only triggers via explicit API call: `POST /jobs/{job_id}/trigger`
- Configuration example:
```python
{
    "job_type": "video",
    "trigger": {
        "mode": "manual"
    }
}
```

**DISABLED (Job-to-job orchestration)**
- No automatic triggering
- Used when one job launches another programmatically
- Enables conditional workflows (e.g., YOLO detection → VLM analysis)

### API Endpoints

**Start a job:**
```bash
POST /jobs/start
Content-Type: application/json

{
    "job_type": "video",
    "prompt": "Describe what you see in the video.",
    "trigger": {"mode": "periodic", "frame_count": 5},
    "continuous": true
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
