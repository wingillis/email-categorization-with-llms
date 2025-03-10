"""
Generate text embeddings and token counts for each email in the emails.parquet file.
"""

import tyro
import tiktoken
import warnings
import numpy as np
import polars as pl
from math import ceil
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from generate_llm_email_labels import clean_html, remove_urls
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


class Config(BaseModel):
    chunks: int = 256
    cluster_mode: Literal["kmeans", "spectral", "agglomerative"] = "kmeans"
    overwrite: bool = False


def _clean_illegal_chars(text):
    return text.encode("utf-8", "ignore").decode("utf-8")


def clean_text(text: str) -> str:
    return clean_html(remove_urls(text))


def count_tokens(text: str, is_cleaned: bool = False) -> int:
    if not is_cleaned:
        text = clean_text(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=set())
    return len(tokens)


def cluster_embeddings(
    df: pl.DataFrame, embedding: str, cluster_mode="spectral"
) -> pl.DataFrame:

    cluster_modes = {
        "spectral": SpectralClustering(n_clusters=11, random_state=0, affinity="nearest_neighbors"),
        "kmeans": KMeans(n_clusters=11, random_state=0),
        "agglomerative": AgglomerativeClustering(
            n_clusters=11, metric="cosine", linkage="average"
        ),
    }
    pcs = PCA(n_components=200).fit_transform(df[embedding].to_numpy())

    clusterer = cluster_modes[cluster_mode]
    labels = clusterer.fit_predict(pcs)

    return df.with_columns(
        pl.Series(f"cluster_labels_{embedding}", labels),
        cluster_mode=pl.lit(cluster_mode),
    )


def main(config: Config):
    MAX_TOKENS = 8192

    df = pl.read_parquet("emails.parquet")

    if not Path("emails_with_tokens_and_embeddings.parquet").exists() or config.overwrite:
        print("Counting tokens in emails to optimize embedding generation speed...")

        # only used to count tokens
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = df.with_columns(
                pl.col("body")
                .map_elements(
                    count_tokens,
                    return_dtype=pl.UInt32,
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
            n_tokens = frame["num_tokens"].to_list()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned_text = map(clean_text, text)
                # truncate cleaned text to MAX_TOKENS
                cleaned_text = [t[:int(len(t) * MAX_TOKENS / (n + 1))] for t, n in zip(cleaned_text, n_tokens)]

                cleaned_text = [
                    f"Subject: {s}\nFrom: {f}\n{t}"
                    for s, f, t in zip(subjects, froms, cleaned_text)
                ]
                # removes some special characters not handled by the tokenizer
                cleaned_text = list(map(_clean_illegal_chars, cleaned_text))

            cleaned_embeddings = [model.encode(cleaned_text) for model in models]

            for i, row in enumerate(frame.iter_rows(named=True)):
                new_df.append(
                    {
                        **row,
                        "snowflake_embedding_clean": cleaned_embeddings[0][i].tolist(),
                        "gte_embedding_clean": cleaned_embeddings[1][i].tolist(),
                    }
                )
        # try to free memory
        del models

        new_df = pl.DataFrame(new_df).with_columns(
            pl.col("snowflake_embedding_clean").cast(
                pl.Array(pl.Float32, cleaned_embeddings[0][0].size)
            ),
            pl.col("gte_embedding_clean").cast(
                pl.Array(pl.Float32, cleaned_embeddings[1][0].size)
            ),
        )
        print("Writing embeddings to parquet...")
        new_df.write_parquet(
            "emails_with_tokens_and_embeddings.parquet",
            compression_level=5,
        )
        print("Checking that embeddings are not all the same...")
        embedding_arr = new_df["snowflake_embedding_clean"].to_numpy()
        embedding_arr = embedding_arr - embedding_arr[[0]]

        if np.allclose(embedding_arr, 0):
            raise ValueError("Embeddings are all the same, something went wrong.")
    else:
        print("Reading existing embeddings...")
        new_df = pl.read_parquet("emails_with_tokens_and_embeddings.parquet")

    # cluster the embeddings
    for embedding in tqdm(["snowflake_embedding_clean", "gte_embedding_clean"]):
        print(f"Clustering {embedding}...")
        new_df = cluster_embeddings(new_df, embedding, config.cluster_mode)

    print("Writing embeddings with cluster labels to parquet...")
    new_df.write_parquet(
        "emails_with_tokens_and_embeddings.parquet",
        compression_level=5,
    )


if __name__ == "__main__":
    config = tyro.cli(Config)

    main(config)
