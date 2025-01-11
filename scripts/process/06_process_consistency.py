import importlib
import numpy as np
import polars as pl
from pathlib import Path
from scipy.stats import entropy
from toolz import groupby, keyfilter

MODEL_DICT = importlib.import_module("01_integrate").MODEL_DICT


def main():

    # read df to merge with consistency output
    metrics_df = pl.read_parquet("merged_metrics.parquet")

    folder = Path("consistency_data")
    files = list(folder.glob("**/*.parquet"))

    files_by_folder = groupby(lambda x: x.parent.name, files)
    files_by_folder = keyfilter(lambda x: x != folder.name, files_by_folder)

    ent_df = []
    keys = ["subject", "date", "from", "body", "supervised_label"]
    for model, files in files_by_folder.items():

        df = pl.read_parquet(files[0])

        consistency_df = []
        for index, _df in df.group_by(keys):
            index_dict = dict(zip(keys, map(pl.lit, index)))
            vcs = _df["primary_category"].value_counts()
            vcs = vcs.with_columns(**index_dict)
            consistency_df.append(vcs)
        consistency_df = pl.concat(consistency_df)

        pivot_consistency = (
            consistency_df.pivot(on="primary_category", values="count", index=keys)
            .fill_null(0)
            .sort("supervised_label")
        )
        x = pivot_consistency[:, 5:].to_numpy()
        y = np.take_along_axis(x, np.argsort(x, axis=1), axis=1)
        y_dist = y.mean(0)[::-1]
        ent = float(entropy(y_dist))
        ent_df.append({"model": MODEL_DICT[model], "entropy": ent})
    ent_df = pl.DataFrame(ent_df)

    merged = metrics_df.join(ent_df, on="model", how="left")
    merged.write_parquet(
        "merged_metrics_consistency.parquet", compression_level=5
    )
    print("Created consistency dataframe")


if __name__ == "__main__":
    main()
