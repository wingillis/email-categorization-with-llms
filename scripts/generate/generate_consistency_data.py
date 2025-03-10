"""
Generates the categorical labels to understand how consistent LLM categorization
is across different models for the same email.
"""

import sys
import tyro
import importlib
import polars as pl
from math import ceil
from pathlib import Path
from ollama import Client
from tqdm.auto import tqdm
from functools import partial
from pydantic import BaseModel
from generate_llm_email_labels import (
    Labels,
    MODELS,
    load_supervised_dataset,
    process_email,
)

sys.path.append(str(Path("scripts/process").absolute()))
MODEL_DICT = importlib.import_module("01_integrate").MODEL_DICT


class Config(BaseModel):
    temperature: float = 0.75
    resamples: int = 30
    n_emails: int = 5 * len(Labels.list())


def sort_by_parameter_count(model_dict):

    reversed_md = {v: k for k, v in MODEL_DICT.items()}
    print("Reading in parameter table")
    with open("parameter_table.txt", "r") as f:
        lines = f.readlines()
    columns = lines[0].split("|")[1:-1]
    columns = [col.strip() for col in columns]
    data = [line.split("|")[1:-1] for line in lines[2:]]
    data = [[d.strip() for d in row] for row in data]

    df = (
        pl.DataFrame(
            data,
            schema=dict(zip(columns, (pl.String, pl.String, pl.Float32))),
            orient="row",
        )
        .rename({"Model name": "model"})
        .drop("Model name (on Ollama)")
    )
    return {
        k: model_dict[k]
        for k in map(reversed_md.get, df.sort("Param. count (B)")["model"].to_list())
    }


def main(config: Config):
    output = Path("consistency_data")

    client = Client(timeout=120)

    df = load_supervised_dataset().sort(["supervised_label", "subject"])
    # sub-sample the dataset
    emails_per_category = ceil(config.n_emails / len(Labels.list()))
    print("Sampling", emails_per_category, "emails per category")
    sampled_df = pl.concat(
        _df.sample(n=min(emails_per_category, len(_df)), seed=5)
        for _df in df.partition_by("supervised_label", maintain_order=True)
    )

    sorted_models = sort_by_parameter_count(MODELS)

    for k, v in tqdm(sorted_models.items(), desc="Model run"):

        # get list of models
        response = client.list()

        # check to see if model_choice is within
        if not any(map(lambda x: x.model == v, response.models)):
            client = Client(timeout=None)
            print(
                f"\nModel {v} not found. Downloading...(this may take a while)"
            )
            client.pull(v)
            client = Client(timeout=120)

        folder = output / k
        file_name = folder / f"{k}_consistency.parquet"

        if file_name.exists() and len(
            pl.read_parquet(file_name)
        ) == config.resamples * len(sampled_df):
            continue

        folder.mkdir(exist_ok=True, parents=True)

        process_with_model = partial(
            process_email, model_choice=v, temperature=config.temperature, client=client
        )

        output_df = []

        for sample_num in tqdm(range(config.resamples), desc="Resampling"):
            frame = pl.DataFrame(
                map(
                    process_with_model,
                    tqdm(
                        sampled_df.iter_rows(named=True),
                        total=len(sampled_df),
                        desc=f"Running sample {sample_num} for {k}",
                        leave=False,
                    ),
                )
            ).with_columns(sample=pl.lit(sample_num))
            output_df.append(frame)
            pl.concat(output_df).write_parquet(file_name)

        output_df = pl.concat(output_df)
        output_df.write_parquet(file_name, compression_level=5)


if __name__ == "__main__":
    config = tyro.cli(Config)

    main(config)
