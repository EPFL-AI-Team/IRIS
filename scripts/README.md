# Scripts

Training and evaluation scripts that run on GPU.

- `vlm/` - Marcus's training scripts (Qwen fine-tuning, projector training)
- `perception/` - Myriam's training scripts (VideoMAE, datasets)

**Run from project root:**

```bash
uv run python scripts/perception/train_videomae.py
```

**Import from package:**

```python
from iris.perception.action_recognition import VideoMAEModel
from iris.vlm.inference import load_model_and_processor
```
