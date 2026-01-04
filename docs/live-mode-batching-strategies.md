# Live Mode Batching Strategies (Future Work)

This document outlines potential congestion-based batching patterns for live mode inference. These are documented for future reference and are **not being implemented** in the current batch inference feature.

## Context

In live mode:
- Frames arrive in real-time from a camera
- Each message contains **multiple images** (unlike analysis mode where segments are separate)
- Latency is critical for user experience
- Batching multiple messages means batching many images at once (VRAM intensive)

## Pattern A: Token Bucket / Leaky Bucket

**Concept**: Regulate the rate of inference requests based on GPU capacity.

```python
class TokenBucket:
    def __init__(self, capacity=10, refill_rate=1.0):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self, tokens=1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

# Usage in live mode:
if not rate_limiter.consume():
    # GPU overloaded, batch frames
    accumulator.add(frame)
    if len(accumulator) >= congestion_batch_size:
        process_batch(accumulator)
else:
    # GPU has capacity, process immediately
    process_single(frame)
```

**Pros**:
- Smooth rate limiting
- Prevents bursts from overwhelming GPU
- Self-regulating based on capacity

**Cons**:
- Adds complexity
- Requires careful tuning of `refill_rate`
- May feel unpredictable to users

---

## Pattern B: Additive Increase / Multiplicative Decrease (TCP-style)

**Concept**: Dynamically adjust batch size based on queue congestion (inspired by TCP congestion control).

```python
class AdaptiveBatcher:
    def __init__(self):
        self.batch_size = 1  # Start with low latency
        self.min_batch = 1
        self.max_batch = 16
        self.queue_threshold = 5

    def adjust_batch_size(self, queue_depth: int):
        """Adjust batch size based on congestion."""
        if queue_depth > self.queue_threshold:
            # Congestion detected: multiplicative increase
            self.batch_size = min(self.max_batch, self.batch_size * 2)
        elif queue_depth < self.queue_threshold // 2:
            # Low congestion: additive decrease
            self.batch_size = max(self.min_batch, self.batch_size - 1)

        return self.batch_size

# Usage:
batch_size = batcher.adjust_batch_size(queue.qsize())
if len(accumulator) >= batch_size:
    process_batch(accumulator)
```

**Pros**:
- Self-tuning, automatically responds to load changes
- Inspired by proven TCP algorithms
- Gradually adapts to optimal batch size

**Cons**:
- Can oscillate between modes
- Needs careful threshold tuning
- May be slow to stabilize

---

## Pattern C: Timeout-based Accumulation (Nagle's Algorithm)

**Concept**: Accumulate frames for a short time window, then batch process (inspired by Nagle's algorithm for packet coalescing).

```python
class TimeoutBatcher:
    def __init__(self, batch_size=4, timeout_ms=50):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.accumulator = []
        self.last_batch_time = time.time()

    async def add_frame(self, frame):
        """Add frame and check if batch ready."""
        self.accumulator.append(frame)

        elapsed_ms = (time.time() - self.last_batch_time) * 1000

        # Batch ready if size reached OR timeout exceeded
        if len(self.accumulator) >= self.batch_size or elapsed_ms >= self.timeout_ms:
            batch = self.accumulator
            self.accumulator = []
            self.last_batch_time = time.time()
            return batch  # Process this batch

        return None  # Not ready yet

# Usage:
batch = await batcher.add_frame(frame)
if batch:
    await process_batch(batch)
```

**Pros**:
- Simple to implement and understand
- Predictable maximum latency (`timeout_ms`)
- Good balance between batching and responsiveness

**Cons**:
- Fixed timeout may not adapt to varying loads
- May add unnecessary latency during low traffic
- Requires tuning timeout value

---

## Pattern D: Hybrid Congestion Control (Recommended)

**Concept**: Combine queue-based mode switching with timeout batching for best of both worlds.

```python
class HybridBatcher:
    def __init__(self):
        self.mode = "single"  # "single" or "batch"
        self.accumulator = []
        self.batch_timeout_ms = 100
        self.last_frame_time = time.time()
        self.queue_threshold = 5
        self.batch_size = 4

    async def process_frame(self, frame, queue_depth: int):
        """Process frame with hybrid strategy."""
        # Check congestion and switch modes
        if queue_depth > self.queue_threshold and self.mode == "single":
            self.mode = "batch"
            logger.info("Switching to batch mode due to congestion")
        elif queue_depth == 0 and self.mode == "batch":
            # Flush any accumulated frames first
            if self.accumulator:
                await self._flush_batch()
            self.mode = "single"
            logger.info("Switching to single-frame mode")

        # Process based on mode
        if self.mode == "single":
            return await process_single(frame)
        else:
            # Batch mode: accumulate and batch
            self.accumulator.append(frame)
            self.last_frame_time = time.time()

            # Batch ready if size reached OR timeout
            elapsed_ms = (time.time() - self.last_frame_time) * 1000
            if len(self.accumulator) >= self.batch_size or elapsed_ms >= self.batch_timeout_ms:
                return await self._flush_batch()

    async def _flush_batch(self):
        """Process accumulated batch."""
        if not self.accumulator:
            return
        batch = self.accumulator
        self.accumulator = []
        return await process_batch(batch)
```

**Pros**:
- Best of both worlds: low latency when possible, high throughput under load
- Adaptive to changing conditions
- Clear mode transitions with logging

**Cons**:
- Most complex implementation
- Needs testing to avoid mode thrashing
- Requires tuning both queue threshold and timeout

---

## Recommendation for Live Mode (Future Implementation)

**Start with Pattern D (Hybrid)** if implementing live mode batching:

### Configuration
- `queue_threshold`: 5 (queue depth to trigger batching)
- `batch_size`: 4 (small batches in batch mode)
- `batch_timeout_ms`: 100 (short timeout to minimize latency)
- Clear logging when switching modes for debugging

### Key Considerations

**VRAM Usage**: Live mode has **multiple images per message** already (unlike analysis mode where each segment is separate). This means:
- Batching in live mode = batching multiple messages (each with multiple images)
- Need to consider total image count across batch, not just message count
- VRAM usage grows faster than analysis mode batching

**Conservative Approach**:
- Max 4 messages per batch in live mode
- If average 8 images/message → 32 images total per batch
- Monitor VRAM with `nvidia-smi` and adjust accordingly

### Testing Strategy

1. **Start with threshold testing**:
   - Test different `queue_threshold` values (3, 5, 10)
   - Measure latency vs throughput tradeoffs

2. **Batch size tuning**:
   - Start with `batch_size=2`
   - Monitor VRAM usage
   - Gradually increase if headroom allows

3. **Timeout tuning**:
   - Test different `batch_timeout_ms` values (50ms, 100ms, 200ms)
   - Measure impact on user-perceived latency

4. **Mode thrashing detection**:
   - Log mode transitions
   - Look for rapid switching (indicates poor threshold tuning)

---

## Alternative: Don't Batch Live Mode

The simplest approach may be to **not batch live mode at all**:
- Live mode is already optimized with frame dropping
- Latency is more critical than throughput for real-time use
- Focus batching optimization on analysis mode where throughput matters
- Keep live mode simple and predictable

This is the approach taken in the initial batch inference implementation.
