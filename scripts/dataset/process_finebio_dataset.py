from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DATA_PATH_BASE = Path("/Users/marcushamelink/Developer/ml/IRIS-semester-project/data")
# DATA_PATH_BASE = Path('/work/team-ai/IRIS_vlm')

# For local
PATHS = {
    "finebio_base": DATA_PATH_BASE / "finebio",
    "finebio_videos": DATA_PATH_BASE / "finebio/videos/w640",
    "finebio_annot": DATA_PATH_BASE / "finebio/annotations/finebio_action_annotations",
    "chuv_base": DATA_PATH_BASE / "colony_counting/DATASET_1",
    "output": DATA_PATH_BASE / "processed",
}


class ChatMessage(BaseModel):
    """Individual message for ChatML"""

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender"
    )
    content: str = Field(description="Text content of the message")


class ChatMLSample(BaseModel):
    """Dictionary format for the ChatML to include messages for an LLM"""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation", min_length=2
    )

    type: str = Field(
        default="chatml", description="Format type, in the case of Qwen, is chatml"
    )

    source: str | None = Field(
        default=None, description="Optional data source metadata for the dataset"
    )


def fill_task_column(details_df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Fill task column for all rows based on time overlap with task annotations"""

    result_df: pd.DataFrame = details_df.copy()

    for _, task_row in tasks_df.iterrows():
        task_start: float = task_row["start_sec"]
        task_end: float = task_row["end_sec"]
        task: str = task_row["task"]

        mask: pd.Series = (
            (result_df["start_sec"] < task_end)
            & (result_df["start_sec"] > task_start)
            & (result_df["end_sec"] < task_end)
            & (result_df["end_sec"] > task_start)
            & (result_df["start_sec"] < result_df["end_sec"])
            & (result_df["task"].isna())
        )

        result_df.loc[mask, "task"] = task

    return result_df


# def extract_frames_uniform(video_path, start_sec, end_sec, max_frames):
#     """
#     Opens video, jumps to specific timestamps, and extracts exactly 'max_frames'.
#     """
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         return []

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     start_f = int(start_sec * fps)
#     end_f = int(end_sec * fps)

#     # Handle edge case where segment is < 1 frame
#     if end_f <= start_f:
#         end_f = start_f + 1

#     # Uniformly pick frame indices
#     indices = np.linspace(start_f, end_f, max_frames, dtype=int)
#     # Deduplicate indices if video segment is too short for 16 distinct frames
#     indices = np.unique(indices)

#     frames = []
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if ret:
#             # OpenCV BGR -> PIL RGB
#             frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

#     cap.release()
#     return frames


def main() -> None:
    # Setup output dirs
    PATHS["output_frames"].mkdir(parents=True, exist_ok=True)

    print("Loading annotation files...")
    txt_files = list(PATHS["annotations"].glob("*.txt"))

    all_dfs = []
    for txt_file in txt_files:
        video_id = txt_file.stem
        try:
            df = pd.read_csv(txt_file)

            # Split into Task-level and Action-level
            task_mask = df["task"].notna()
            task_df = df[task_mask]
            action_df = df[~task_mask]

            # Map tasks to actions
            processed_df = fill_task_column(action_df, task_df)
            processed_df["video_id"] = video_id
            all_dfs.append(processed_df)

        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")

    if not all_dfs:
        print("No valid annotations found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Remove rows where we still don't know the task (optional, depending on your needs)
    final_df = final_df[final_df["task"].notna()]

    print(f"Total samples to process: {len(final_df)}")


# txt_files = list(PATHS["finebio_annot"].glob("*.txt"))
# # print(f"Found {len(txt_files)} text files")


# # output_data: list[ChatMLSample] = []
# all_dfs_collected: list[pd.DataFrame] = []

# TXT_FILE_LIMIT = 1

# for txt_file in islice(txt_files, TXT_FILE_LIMIT):
#     video_id = txt_file.stem
#     # print(video_id, txt_file)

#     df = pd.read_csv(txt_file)

#     # Get rows with task not None
#     task_mask: pd.Series = df["task"].notna()
#     task_df: pd.DataFrame = df[task_mask]
#     main_df: pd.DataFrame = df[~task_mask]

#     main_df_filled: pd.DataFrame = fill_task_column(main_df, task_df)

#     # Prepare for creating CSV entries, add video ID
#     main_df_filled = main_df_filled.assign(video_id=video_id)

#     df["segment_duration"] = df["end_sec"] - df["start_sec"]
#     print(df["segment_duration"].describe())
#     all_dfs_collected.append(main_df_filled)


# final_df = pd.concat(all_dfs_collected, ignore_index=True)
# output_file = PATHS["output"] / "output.csv"
# final_df.to_csv(output_file, index=False)

main()
