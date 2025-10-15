# IRIS Project - Typing Philosophy & Standards

## TL;DR: Progressive Typing Strategy

**Notebooks → Scripts → Production**: Type more as code moves toward production.

## Typing Levels by File Type

### Level 0: Notebooks (Minimal Typing)

**Location**: `notebooks/`  
**Rule**: Type only when it clarifies ambiguous variables

```python
# ✅ DO: Type constants and config
batch_size: int = 32
learning_rate: float = 1e-4
model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

# ❌ SKIP: Loop variables, obvious types
for image in images:  # Don't type
    result = model(image)  # Don't type
```

### Level 1: Root Scripts (Function Signatures)

**Location**: `main.py`, runner scripts  
**Rule**: Type all function inputs and outputs

```python
# ✅ DO: Type function signatures
def load_model_and_processor(
    model_key: str
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model from config."""
    ...

# ⚠️ OPTIONAL: Type intermediate variables
def process_frame(frame: np.ndarray) -> Image.Image:
    # Internal vars don't need types
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(converted)
```

### Level 2: Source Code (Moderate Typing)

**Location**: `src/`  
**Rule**: Type all public APIs, skip complex ML internals

```python
# ✅ DO: Type public functions fully
def preprocess_images(
    images: list[Image.Image],
    target_size: tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Preprocess images for model input."""
    # Internal operations can skip types
    processed = []
    for img in images:
        resized = img.resize(target_size)
        processed.append(resized)
    return torch.stack(processed)

# ⚠️ SKIP: Complex tensor operations
def compute_attention(q, k, v):  # Tensor shapes are too complex
    scores = torch.matmul(q, k.transpose(-2, -1))
    return torch.softmax(scores, dim=-1)
```

### Level 3: Production/Library Code (Full Typing)

**Location**: Shared utilities, APIs, libraries  
**Rule**: Everything typed, use mypy strict

```python
from typing import Protocol, TypeVar, Generic

T = TypeVar('T')

class VisionModel(Protocol):
    """Protocol for vision models."""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def generate(self, **kwargs: Any) -> torch.Tensor: ...

class InferenceQueue(Generic[T]):
    """Type-safe inference queue."""
    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
    
    async def put(self, item: T) -> None:
        await self._queue.put(item)
```

## Specific Type Aliases for ML

Use these consistently across the project:

```python
# Vision types
from PIL import Image
import numpy as np
import torch

# Type aliases (define in src/types.py or similar)
ImageType = Image.Image
NumpyImage = np.ndarray  # Shape: (H, W, C)
TensorImage = torch.Tensor  # Shape: (C, H, W) or (B, C, H, W)

# Transformers types
from transformers import PreTrainedModel, ProcessorMixin
ModelType = PreTrainedModel
ProcessorType = ProcessorMixin
```

## When to Use `Any`

```python
from typing import Any

# ✅ DO: Use Any for truly dynamic content
def process_model_output(output: Any) -> dict[str, float]:
    """Model outputs vary too much to type strictly."""
    ...

# ❌ AVOID: Using Any out of laziness
def process_image(img: Any) -> Any:  # Too lazy!
    # Should be: (img: Image.Image) -> torch.Tensor
    ...
```

## Ruff Configuration

Your `ruff.toml` enforces this philosophy:

```toml
[lint]
select = ["ANN"]  # Type annotation rules enabled

[lint.per-file-ignores]
"notebooks/*.py" = ["ANN"]           # No types required
"notebooks/*.ipynb" = ["ANN"]        # No types required
"main.py" = ["ANN001", "ANN002", "ANN003"]  # Args optional
"src/**/*.py" = []                   # Full typing enforced
"src/models/*.py" = ["ANN001"]       # Complex tensor args optional
```

## Transformers-Specific Typing

### Model Loading

```python
def load_vision_model(
    model_id: str,
    device: str | torch.device = "auto"
) -> PreTrainedModel:
    """Load vision-language model."""
    return AutoModelForVision2Seq.from_pretrained(
        model_id, device_map=device
    )
```

### Forward Pass (Complex - Type What Matters)

```python
def run_inference(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    image: Image.Image,
    prompt: str,
) -> str:
    """Run model inference."""
    # Don't type every intermediate tensor
    inputs = processor(images=[image], text=[prompt], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    result: str = processor.batch_decode(outputs)[0]
    return result
```

## Documentation References

Always check official docs when uncertain:

- **Transformers Types**: [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- **PyTorch Types**: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- **Python Typing**: [Python typing module](https://docs.python.org/3/library/typing.html)
- **Asyncio Types**: [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

Use Context7 tool for library-specific documentation in your IDE.

## For LLMs Working on This Project

When generating code for IRIS:

1. **Check file location** - apply appropriate typing level
2. **Type function signatures** - always for `src/`, main runners
3. **Skip obvious types** - loop vars, simple assignments
4. **Use established aliases** - `ImageType`, `ModelType`, etc.
5. **Reference docs** - check HuggingFace/PyTorch docs when unsure
6. **Add docstrings** - brief one-liners for all functions

## Quick Reference Table

| Location      | Type Level | Function Args  | Function Returns | Internals       |
| ------------- | ---------- | -------------- | ---------------- | --------------- |
| `notebooks/`  | Minimal    | ❌              | ❌                | ❌               |
| `main.py`     | Light      | ✅              | ✅                | ❌               |
| `src/`        | Moderate   | ✅              | ✅                | ⚠️ (public only) |
| `src/models/` | Moderate   | ⚠️ (complex ok) | ✅                | ❌               |
| Libraries     | Full       | ✅              | ✅                | ✅               |

**Legend**: ✅ = Required, ⚠️ = Optional/Selective, ❌ = Skip

---

Last updated: October 2025  
Philosophy: Pragmatic typing for student research - rigor without bureaucracy
