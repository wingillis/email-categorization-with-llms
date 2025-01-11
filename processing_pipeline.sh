#!/bin/bash
# must be run in this project's root directory

set -e

uv run python scripts/process/00_count_raw_tokens.py

# 1. combine dataframes per-model
uv run python scripts/process/01_integrate.py --folder structured_output
uv run python scripts/process/01_integrate.py --folder structured_output_supervised

# 2. create failure rate and inference duration dataframes
uv run python scripts/process/02_failure_rate_and_inference_speed.py

# 3. merge supervised data across all models
uv run python scripts/process/03_merge_supervised_data.py

# 4. compute supervised metrics and merge with inference speed df
PYTHONWARNINGS=ignore uv run python scripts/process/04_compute_metrics_and_merge.py

# 5. process parameter table and merge
uv run python scripts/process/05_process_parameter_table.py

# 6. process classification consistency and merge
uv run python scripts/process/06_process_consistency.py