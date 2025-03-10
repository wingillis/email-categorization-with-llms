import polars as pl
from pathlib import Path


def main():
    files = list(Path(".").glob("structured_output*/*.parquet"))
    supervised_files = Path(".").glob("training_dataset*label*.parquet")

    supervised_df = (
        pl.concat(
            pl.scan_parquet(file)
            .drop_nulls(subset=["supervised_label"])
            .select(["from", "body", "date", "subject", "supervised_label"])
            for file in supervised_files
        )
        .with_columns(
            pl.col("supervised_label").str.replace(r"\(.*\)", "").str.strip_chars(),
        )
        .collect()
    )

    assert len(supervised_df) > 0, "No supervised data found"

    joined_df = supervised_df.clone()

    for file in files:
        df = (
            pl.scan_parquet(file)
            .select(["from", "body", "date", "subject", "primary_category", "model"])
            .collect()
        )
        model = df["model"][0]
        joined_df = joined_df.join(
            df.drop("model"), on=["from", "body", "date", "subject"], how="left"
        ).rename({"primary_category": model})
    joined_df.write_parquet(
        "merged_supervised_data.parquet", compression_level=5
    )


if __name__ == "__main__":
    main()
