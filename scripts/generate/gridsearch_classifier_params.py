import tyro
import warnings
import numpy as np
import polars as pl
from typing import Literal
from tqdm.auto import tqdm
from sklearn.svm import SVC
from itertools import product
from pydantic import BaseModel
from toolz import keymap, merge
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    RepeatedStratifiedKFold,
)


class Config(BaseModel):
    llm_name: Literal["tulu", "granite"] = "tulu"
    samples_per_cat: int = 200


CLASSIFIERS = {
    "svc": SVC(probability=False, class_weight="balanced", tol=1e-2),
    "knn": KNeighborsClassifier(n_jobs=1),
    "rf": RandomForestClassifier(class_weight="balanced", n_jobs=1),
    "gnb": GaussianNB(),
    "lr": LogisticRegression(
        class_weight="balanced",
        solver="saga",
        n_jobs=1,
        max_iter=5_000,
        tol=1e-1,
    ),
}

CLASSIFIER_PARAMS = {
    "svc": {
        "kernel": ["linear", "rbf"],
        "C": [1e-2, 0.1, 0.9],
    },
    "knn": {
        "n_neighbors": [5, 15, 50, 100],
        "weights": ["distance"],
        "metric": ["euclidean", "cosine", "manhattan"],
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 20, None],
    },
    "gb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 20, None],
    },
    "lr": {
        "C": [1e-2, 0.1, 0.9],
        "penalty": ["l1", "l2"],
    },
}

PCA_PARAMS = {
    "n_components": [25, 50, 250, 400],
}


# sample items so that items in each category are reasonably distant from each other
# - adds robustness to the model
def get_sample(df: pl.DataFrame, n=100, random_state=0) -> pl.DataFrame:
    """Run k-Center algorithm to select most diverse samples for a category."""
    if len(df) <= n:
        return df

    # embedding is hard-coded for reproducibility and interpretable comparisons
    distance_mtx = pdist(df["snowflake_embedding_clean"].to_numpy(), metric="cosine")
    distance_mtx = squareform(distance_mtx)

    rng = np.random.default_rng(random_state)

    sample = rng.integers(0, len(df), size=1).tolist()
    dist_row = distance_mtx[sample[0]]

    for _ in range(n - 1):
        idxes = np.argsort(dist_row)
        # make a bit robust, use quantile
        pos = int(len(idxes) * 0.95)

        idx = idxes[pos]
        while idx in sample and pos < (len(df) - 1):
            pos += 1
            idx = idxes[pos]

        while idx in sample and pos >= 0:
            pos -= 1
            idx = idxes[pos]

        sample.append(idx)
        dist_row = np.minimum(dist_row, distance_mtx[sample[-1]])

    return df[sample]


def load_dataframe(llm_name: str) -> pl.DataFrame:

    if llm_name == "granite":
        # granite
        df = pl.scan_parquet(
            "structured_output/granite3_1-dense-8b_structured_output_df.parquet"
        )
    elif llm_name == "tulu":
        # tulu
        df = pl.scan_parquet("structured_output/tulu3_structured_output_df.parquet")

    # merge with embedding df
    df = df.join(
        pl.scan_parquet("emails_with_tokens_and_embeddings.parquet").select(
            [
                "subject",
                "from",
                "body",
                "date",
                "snowflake_embedding_clean",
                "gte_embedding_clean",
            ]
        ),
        on=["subject", "from", "body", "date"],
        how="left",
    ).collect()

    df = df.drop_nulls(subset=["snowflake_embedding_clean", "gte_embedding_clean"])
    df = df.filter(pl.col("snowflake_embedding_clean").arr.sum() != 0).filter(
        pl.col("primary_category") != "N/A"  # came from CUDA error
    )

    print("Loaded dataframe with shape:", df.shape)

    return df


def run_grid_search(X, y, pipeline, model_name):

    model_params = CLASSIFIER_PARAMS.get(model_name, {})
    model_params = keymap(lambda k: f"model__{k}", model_params)
    model_params = merge(model_params, keymap(lambda k: f"pca__{k}", PCA_PARAMS))

    grid_search = GridSearchCV(
        pipeline,
        model_params,
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
        n_jobs=6,
        verbose=1,
        scoring="f1_weighted",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X, y)

    return pl.DataFrame(grid_search.cv_results_, strict=False).sort("rank_test_score")


def main(config: Config):

    df = load_dataframe(config.llm_name)

    with pl.Config(tbl_rows=20):
        print(df["primary_category"].value_counts(sort=True))

    sample = pl.concat(
        [
            get_sample(_df, n=config.samples_per_cat, random_state=0)
            for _df in tqdm(
                df.partition_by("primary_category", maintain_order=True),
                desc="Sampling categories",
            )
        ]
    )

    embeddings = ["gte_embedding_clean", "snowflake_embedding_clean"]
    combos = list(product(embeddings, CLASSIFIERS.items()))

    results = {}
    labels = sample["primary_category"].to_numpy()

    for embedding_name, (model_name, model) in tqdm(
        combos, desc="Grid-searching model archs"
    ):
        print(f"Training {model_name} on {embedding_name}")
        pipeline = Pipeline(
            [("pca", PCA()), ("scaler", StandardScaler()), ("model", model)]
        )

        embedding = sample[embedding_name].to_numpy()

        results_df = run_grid_search(embedding, labels, pipeline, model_name)
        results[(embedding_name, model_name)] = results_df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_results = []
        for (embedding_name, model_name), v in results.items():
            _df = (
                v.select(["mean_test_score", "params", "rank_test_score"])
                .head(10)
                .with_columns(
                    pl.col("params").map_elements(str, return_dtype=pl.String),
                    embedding=pl.lit(embedding_name),
                    model=pl.lit(model_name),
                    llm=pl.lit(config.llm_name),
                )
            )
            combined_results.append(_df)
        combined_results = pl.concat(combined_results).sort(
            "mean_test_score", descending=True
        )

    combined_results.write_parquet(
        f"classifier_grid_search/{config.llm_name}_model_selection_results.parquet"
    )

    print("Computing accuracy for top results per model")
    top_results = combined_results.filter(pl.col("rank_test_score") == 1)

    output = []
    for result in top_results.iter_rows(named=True):

        best_params = eval(result["params"])

        pipe = Pipeline(
            [
                ("pca", PCA()),
                ("scaler", StandardScaler()),
                ("model", CLASSIFIERS[result["model"]]),
            ]
        )
        pipe = pipe.set_params(**best_params)

        x = cross_validate(
            pipe,
            sample[result["embedding"]].to_numpy(),
            labels,
            cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=0),
            n_jobs=6,
            scoring=["f1_weighted", "accuracy"],
        )
        for i, (f1, acc) in enumerate(zip(x["test_f1_weighted"], x["test_accuracy"])):
            output.append(merge(result, dict(f1=f1, accuracy=acc, iteration=i)))

    output_df = pl.DataFrame(output)
    output_df.write_parquet(
        f"classifier_grid_search/{config.llm_name}_top_results_per_model.parquet",
        compression_level=5,
    )

    print("Computing accuracy for top results per embedding")
    # compare embeddings for SVC
    result = top_results.sort("mean_test_score", descending=True).to_dicts()[0]

    print("Best result:", result)
    best_params = eval(result["params"])

    pipe = Pipeline(
        [
            ("pca", PCA()),
            ("scaler", StandardScaler()),
            ("model", CLASSIFIERS[result["model"]]),
        ]
    )
    pipe = pipe.set_params(**best_params)

    output = []
    for embedding_name in embeddings:
        x = cross_validate(
            pipe,
            sample[embedding_name].to_numpy(),
            labels,
            cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=0),
            n_jobs=6,
            scoring=["f1_weighted", "accuracy"],
        )

        for i, (f1, acc) in enumerate(zip(x["test_f1_weighted"], x["test_accuracy"])):
            output.append(
                merge(
                    result,
                    dict(f1=f1, accuracy=acc, iteration=i, embedding=embedding_name),
                )
            )

    output_df = pl.DataFrame(output)
    output_df.write_parquet(
        f"classifier_grid_search/{config.llm_name}_top_model_results_per_embedding.parquet",
        compression_level=5,
    )


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
