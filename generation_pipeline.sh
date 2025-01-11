# Note that this script isn't intended to actually be run.
# It serves as documentation for the order of operations for generating the datasets.
set -e

# download emails
uv run python scripts/generate/create_email_df.py

# create embeddings
uv run python scripts/generate/generate_embeddings.py

# generate llm-based labels
uv run python scripts/generate/generate_llm_email_labels.py

# generate consistency data
uv run python scripts/generate/generate_consistency_data.py

# TODO: make script that greates the hand-labeling dataset
# ... insert here ...

# hand-label some data
uv run python scripts/generate/create_supervised_dataset.py