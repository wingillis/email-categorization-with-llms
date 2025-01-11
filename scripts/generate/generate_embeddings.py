"""
Generate text embeddings and token counts for each email in the emails.parquet file.
"""

import tyro
import tiktoken
import warnings
import numpy as np
import polars as pl
from math import ceil
from tqdm.auto import tqdm
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from generate_llm_email_labels import clean_html, remove_urls


class Config(BaseModel):
    chunks: int = 256


def clean_text(text: str) -> str:
    return clean_html(remove_urls(text))


def count_tokens(text: str, is_cleaned: bool = False) -> int:
    if not is_cleaned:
        text = clean_text(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=set())
    return len(tokens)


def main(config: Config):
    df = pl.read_parquet("emails.parquet")

    print("Counting tokens in emails to optimize embedding generation speed...")

    # only used to count tokens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.with_columns(
            pl.col("body")
            .map_elements(
                count_tokens,
                return_dtype=pl.UInt32,
                strategy="threading",
            )
            .alias("num_tokens")
        )
    # speed up embedding generation by batching similar length emails
    df = df.sort("num_tokens")

    # context size for this model is 8192
    models = [
        SentenceTransformer(
            "Snowflake/snowflake-arctic-embed-m-v2.0",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "bfloat16"},
        ),
        SentenceTransformer(
            "Alibaba-NLP/gte-base-en-v1.5",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "bfloat16"},
        ),
    ]

    new_df = []

    for frame in tqdm(
        df.iter_slices(n_rows=config.chunks),
        total=ceil(len(df) / config.chunks),
        desc="Embedding chunks",
    ):
        text = frame["body"].to_list()
        subjects = frame["subject"].to_list()
        froms = frame["from"].to_list()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned_text = map(clean_text, text)
            cleaned_text = [
                f"Subject: {s}\nFrom: {f}\n{t}"
                for s, f, t in zip(subjects, froms, cleaned_text)
            ]

        try:
            cleaned_embeddings = [model.encode(cleaned_text) for model in models]
        except TypeError:
            cleaned_embeddings = [[], []]

            for text in cleaned_text:
                for mdl_num, model in enumerate(models):
                    try:
                        cleaned_embeddings[mdl_num].append(model.encode(text))
                    except UnicodeEncodeError:
                        cleaned_embeddings[mdl_num].append(np.zeros(768))
                        print(text)
            cleaned_embeddings = [np.array(x) for x in cleaned_embeddings]
            print(cleaned_embeddings[0].shape)

        for i, row in enumerate(frame.iter_rows(named=True)):
            new_df.append(
                {
                    **row,
                    "snowflake_embedding_clean": cleaned_embeddings[0][i].tolist(),
                    "gte_embedding_clean": cleaned_embeddings[1][i].tolist(),
                }
            )

    new_df = pl.DataFrame(new_df).with_columns(
        pl.col("snowflake_embedding_clean").cast(
            pl.Array(pl.Float32, cleaned_embeddings[0][0].size)
        ),
        pl.col("gte_embedding_clean").cast(
            pl.Array(pl.Float32, cleaned_embeddings[1][0].size)
        ),
    )
    new_df.write_parquet(
        "emails_with_tokens_and_embeddings.parquet",
        compression_level=5,
    )


if __name__ == "__main__":
    config = tyro.cli(Config)

    main(config)
