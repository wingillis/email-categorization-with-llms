# Email Categorization project

This is the code that goes with the [companion blog post](https://wingillis.github.io/posts/2025-02-17-email-categorization/).

## Setup

Make sure ollama is installed separately.

Python code for this project can be installed with `uv`:

```bash
cd email-categorization-with-llms
# creates a .venv with packages from uv.lock and pyproject.toml
uv sync
```

The email dataframe generation code requires a `credentials.toml` file with your IMAP credentials:
```toml
email = "your-email@email.com"
password = "imap-password"
```

## Running the generation pipeline

The generation pipeline creates the email dataframe, embeddings, and llm-based labels.
The `generation_pipeline.sh` script can be used as a reference on how to execute the pipeline.

## Running the processing pipeline

The processing pipeline creates dataframes for plotting and analyzing the results.
The `processing_pipeline.sh` script can be directly executed to run the pipeline:

```bash
# run this in the root directory of the project
bash processing_pipeline.sh
```
