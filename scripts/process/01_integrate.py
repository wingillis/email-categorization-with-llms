"""
Combines dataframes saved during LLM structured output generation into a single dataframe.
Adds model name and lightly processes the categories to remove the parenthetical component.
"""

import tyro
import polars as pl
from pathlib import Path
from pydantic import BaseModel
from toolz import groupby, keyfilter


class Config(BaseModel):
    folder: Path = Path.cwd()


BLACKLIST = [
    "gemma2-2b",
    "phi3_5",
]


MODEL_DICT = {
    "command-r7b": "Command-R-7b",
    "dolphin3": "Dolphin-3",
    "falcon3_3b": "Falcon-3-3b",
    "falcon3_7b": "Falcon-3-7b",
    "granite3_1-dense-2b": "Granite-3.1-2b",
    "granite3_1-dense-8b": "Granite-3.1-8b",
    "internlm3": "InternLM-3",
    "llama3_1": "Llama-3.1-8b",
    "llama3_2": "Llama-3.2-3b",
    "marco-o1": "Marco-o1",
    "nemo": "Nemo",
    "olmo2": "Olmo-2",
    "qwen-1_5b": "Qwen-2.5-1.5b",
    "qwen-3b": "Qwen-2.5-3b",
    "qwen-7b": "Qwen-2.5-7b",
    "smallthinker": "Smallthinker",
    "tulu3": "Tulu-3",
    "deepseek-r1_1.5b": "DeepSeek-r1-distill-1.5b",
    "deepseek-r1_7b": "DeepSeek-r1-distill-7b",
    "deepseek-r1_8b": "DeepSeek-r1-distill-8b",
    "gemma2": "Gemma-2",
    "llama3.3": "Llama-3.3",
    "phi4": "Phi-4",
    "qwen-32b": "Qwen-2.5-32b",
    "qwq": "QwQ",
    "qwen-0_5b": "Qwen-2.5-0.5b",
    "granite3_1-moe-1b": "Granite-3.1-MOE-1b",
    "granite3_1-moe-3b": "Granite-3.1-MOE-3b",
    "mistral-small": "Mistral-small-3",
    "smollm2": "SmolLM-2-1.7b",
    "smollm2-360m": "SmolLM-2-360m",
}


def blacklist(d: dict, keys: list | tuple):
    return keyfilter(lambda x: x not in keys, d)


def main(config: Config):
    folder = config.folder
    if config.folder == Path.cwd():
        folder = config.folder / f"structured_output"
    files = list(folder.glob("**/*.parquet"))

    files_by_folder = groupby(lambda x: x.parent.name, files)
    files_by_folder = blacklist(files_by_folder, BLACKLIST)
    files_by_folder = keyfilter(lambda x: x != folder.name, files_by_folder)

    category_column = "primary_category"

    for model, _files in files_by_folder.items():
        print(f"Processing {model}")
        model_dfs = []
        for file in _files:
            df = (
                pl.scan_parquet(file)
                .with_columns(
                    pl.col("date").cast(pl.Datetime("us", "UTC")),
                    pl.col(category_column)
                    .str.replace(r"\(.*\)", "")
                    .str.strip_chars()
                    .str.replace(r"^na$", "N/A"),
                )
                .filter(pl.col("date") > pl.datetime(1991, 1, 1, time_zone="UTC"))
                .with_columns(model=pl.lit(MODEL_DICT[model]))
                .collect()
            )
            model_dfs.append(df)
        col_order = model_dfs[0].columns
        df = pl.concat((_df[col_order] for _df in model_dfs), how="vertical_relaxed")
        df.write_parquet(
            folder / f"{model}_structured_output_df.parquet", compression_level=4
        )
        print(f"Done processing {model}")


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
