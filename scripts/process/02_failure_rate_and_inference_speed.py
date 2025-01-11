import polars as pl
from pathlib import Path


def main():
    files = list(Path(".").glob("structured_output/*.parquet"))

    processed_dfs = []
    for file in files:
        print(f"Processing {file}")
        df = pl.scan_parquet(file)
        processed_dfs.append(
            df.group_by("model")
            .agg(
                ((pl.col("primary_category") == "N/A").mean() * 100).alias(
                    "Failure rate (%)"
                ),
                pl.col("inference_duration").median().alias("Inference duration (s)"),
            )
            .collect()
        )
    processed_dfs = pl.concat(processed_dfs)
    processed_dfs.write_parquet(
        "failure_rate_inference_dur.parquet", compression_level=5
    )
    print(processed_dfs)


if __name__ == "__main__":
    main()
