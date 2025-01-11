import polars as pl
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score


def main():

    df = pl.read_parquet("merged_supervised_data.parquet")
    models = df.columns[5:]

    print("Computing accuracy and f1 score for models")

    acc_f1_df = []
    for model in models:
        x = df.drop_nulls(subset=[model])
        acc_f1_df.append(
            dict(
                model=model,
                accuracy=balanced_accuracy_score(x["supervised_label"], x[model]),
                f1=f1_score(x["supervised_label"], x[model], average="weighted"),
                precision=precision_score(
                    x["supervised_label"], x[model], average="weighted"
                ),
            )
        )
    acc_f1_df = pl.DataFrame(acc_f1_df)

    # load the failure rate and inference duration data
    fr_inf_df = pl.read_parquet("failure_rate_inference_dur.parquet")

    # merge the dataframes
    merged = acc_f1_df.join(fr_inf_df, on="model", how="left")
    merged.write_parquet(
        "merged_metrics.parquet", compression_level=5
    )
    print("Merged metrics saved")


if __name__ == "__main__":
    main()
