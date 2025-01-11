import tiktoken
import polars as pl


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Counts tokens in a text string using tiktoken.

    Args:
        text (str): The text to tokenize.
        model_name (str, optional): The encoding model to use.
                                     Defaults to "cl100k_base" (used by GPT-3 and GPT-4).

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(model_name)
    tokens = encoding.encode(text, disallowed_special=set())
    return len(tokens)


def main():
    df = pl.scan_parquet("emails.parquet")
    # only used to count tokens

    df = df.with_columns(
        pl.col("body")
        .map_elements(
            count_tokens,
            return_dtype=pl.UInt32,
            strategy="threading",
        )
        .alias("num_raw_tokens")
    )

    df2 = pl.scan_parquet("emails_with_tokens_and_embeddings.parquet").select(
        ["body", "date", "from", "subject", "num_tokens"]
    )

    combined = df.join(df2, on=["body", "date", "from", "subject"], how="inner")

    stats = combined.select(
        pl.col("num_tokens").mean().alias("Avg. cleaned token count"),
        pl.col("num_raw_tokens").mean().alias("Avg. raw token count"),
        pl.col("num_tokens").quantile(0.995).alias("99.5 percentile cleaned token count"),
        pl.col("num_raw_tokens").quantile(0.995).alias("99.5 percentile raw token count"),
    ).with_columns(pl.all().cast(pl.UInt32)).collect()

    print(stats)


if __name__ == "__main__":
    main()
