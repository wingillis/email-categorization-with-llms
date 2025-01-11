import polars as pl


def main():
    print("Reading in parameter table")
    with open("parameter_table.txt", "r") as f:
        lines = f.readlines()
    columns = lines[0].split("|")[1:-1]
    columns = [col.strip() for col in columns]
    data = [line.split("|")[1:-1] for line in lines[2:]]
    data = [[d.strip() for d in row] for row in data]

    df = pl.DataFrame(
        data,
        schema=dict(zip(columns, (pl.String, pl.String, pl.Float32))),
        orient="row",
    ).rename({"Model name": "model"}).drop("Model name (on Ollama)")

    # read the metrics dataframe
    metrics_df = pl.read_parquet("merged_metrics.parquet")

    # join the dataframes
    merged = df.join(metrics_df, on="model", how="right")
    merged.write_parquet("merged_metrics.parquet")
    print("Merged with metrics dataframe - overwriting")


if __name__ == "__main__":
    main()
