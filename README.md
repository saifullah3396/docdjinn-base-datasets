# DocDjinn Base Datasets

Download and preprocess the document datasets used by DocDjinn into a local cache.

## Setup
Install [uv](https://docs.astral.sh/uv/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Prepare a single dataset
Datasets are addressed as `<dataset>/<config>` (see `DATASETS` in [catalog.py](src/atria_datasets/catalog.py)).
Output is written to `<data_dir>/<dataset>/storage/<config>/`.
```bash
uv run python -m atria_datasets.prepare_dataset funsd/default /path/to/datasets
```
Useful flags: `--max_samples=50` (quick test; use a separate data dir), `--overwrite_existing_cached=True`, `--print_samples=False`.

## Prepare multiple datasets
[examples/prepare_dataset.sh](examples/prepare_dataset.sh) runs the predefined groups, or any single entry:
```bash
bash examples/prepare_dataset.sh small_datasets /path/to/datasets
bash examples/prepare_dataset.sh big_datasets /path/to/datasets
```

## Load a prepared dataset
```python
from atria_datasets import AtriaDataset

dataset = AtriaDataset.load_by_name("funsd/default", data_dir="/path/to/datasets/funsd")
for sample in dataset.train:
    print(sample)
    break
```
