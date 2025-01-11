"""
Generates the categorical labels to understand how consistent LLM categorization
is across different models for the same email.
"""

import tyro
import polars as pl
from math import ceil
from pathlib import Path
from ollama import Client
from tqdm.auto import tqdm
from functools import partial
from pydantic import BaseModel
from email_struct_output import Labels, MODELS, load_supervised_dataset, process_email


class Config(BaseModel):
    temperature: float = 0.75
    resamples: int = 30
    n_emails: int = 5 * len(Labels.list())


def main(config: Config):
    output = Path("consistency_data")

    client = Client(timeout=50)

    df = load_supervised_dataset().sort(["supervised_label", "subject"])
    # sub-sample the dataset
    emails_per_category = ceil(config.n_emails / len(Labels.list()))
    print("Sampling", emails_per_category, "emails per category")
    sampled_df = pl.concat(
        _df.sample(n=min(emails_per_category, len(_df)), seed=5)
        for _df in df.partition_by("supervised_label", maintain_order=True)
    )
    for k, v in tqdm(MODELS.items(), desc="Model run"):
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
