import asyncio
from concurrent.futures import ThreadPoolExecutor

from transformers import PreTrainedModel, ProcessorMixin


class InferenceQueue:
    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        max_queue_size: int = 30,
    ):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        # Apparently GPU inference blocks, so we need an executor

    # async def submit
