from itertools import islice
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


txt_files = list(PATHS["finebio_annot"].glob("*.txt"))
# print(f"Found {len(txt_files)} text files")


# output_data: list[ChatMLSample] = []
all_dfs_collected: list[pd.DataFrame] = []

TXT_FILE_LIMIT = 1

for txt_file in islice(txt_files, TXT_FILE_LIMIT):
    video_id = txt_file.stem
    # print(video_id, txt_file)

    df = pd.read_csv(txt_file)

    # Get rows with task not None
    task_mask: pd.Series = df["task"].notna()
    task_df: pd.DataFrame = df[task_mask]
    main_df: pd.DataFrame = df[~task_mask]

    main_df_filled: pd.DataFrame = fill_task_column(main_df, task_df)

    # Prepare for creating CSV entries, add video ID
    main_df_filled = main_df_filled.assign(video_id=video_id)

    df["segment_duration"] = df["end_sec"] - df["start_sec"]
    print(df["segment_duration"].describe())
    all_dfs_collected.append(main_df_filled)


final_df = pd.concat(all_dfs_collected, ignore_index=True)
output_file = PATHS["output"] / "output.csv"
final_df.to_csv(output_file, index=False)

# main()
