from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _target_frame_slots(
    *, canonical_max_frames: int, frames_per_segment: int
) -> np.ndarray:
    """Return the canonical slot indices to materialize on disk.

    Example: canonical_max_frames=16, frames_per_segment=4 -> [0, 5, 10, 15]
    """
    if canonical_max_frames <= 0:
        raise ValueError("canonical_max_frames must be > 0")
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be > 0")
    if frames_per_segment > canonical_max_frames:
        raise ValueError("frames_per_segment cannot exceed canonical_max_frames")

    slots = np.linspace(0, canonical_max_frames - 1, frames_per_segment, dtype=int)
    return np.unique(slots)


def fill_task_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    FineBio txt files mix 'tasks' (high-level steps) and 'atomic actions'
    in the same list. Tasks usually have empty verbs.
    This maps the 'task' label to all atomic actions that fall within its time window.
    """
    task_rows = df[df["task"].notna() & (df["task"] != "")].copy()
    action_rows = df[df["verb"].notna() & (df["verb"] != "")].copy()

    if task_rows.empty:
        return action_rows

    task_rows = task_rows.sort_values("start_sec")
    action_rows["context_task"] = None

    for _, t_row in task_rows.iterrows():
        t_start, t_end = t_row["start_sec"], t_row["end_sec"]
        t_label = t_row["task"]

        mask = (action_rows["start_sec"] >= t_start) & (
            action_rows["start_sec"] < t_end
        )
        action_rows.loc[mask, "context_task"] = t_label

    return action_rows


def is_valid_action(
    row: Any, min_duration: float = 0.5, max_duration: float = 3.0
) -> bool:
    """Return True if the row describes a valid atomic action.

    Checks:
    - duration (end_sec - start_sec) between min_duration and max_duration
    - verb is present (not NaN)
    - manipulated_object is present and not the string 'nan'
    """
    try:
        start = float(row.get("start_sec", 0.0))
        end = float(row.get("end_sec", 0.0))
    except Exception:
        return False

    duration = end - start
    if duration < min_duration or duration > max_duration:
        return False

    verb = row.get("verb")
    if pd.isna(verb) or str(verb).strip() == "":
        return False

    obj = row.get("manipulated_object")
    if obj is None:
        return False
    ostr = str(obj).strip().lower()
    if ostr == "" or ostr == "nan":
        return False

    return True
