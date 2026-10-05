#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a small_datasets=(
    "tobacco3482/image_with_ocr"
    "cord/default"
    "funsd/default"
    "sroie/default"
    "wild_receipts/default"
    "docile/default"
    "due_benchmark/ExDocVQA"
    "due_benchmark/ExDeepForm"
    "due_benchmark/ExTabFact"
    "due_benchmark/ExWikiTableQuestions"
    "due_benchmark/ExInfographicsVQA"
    "due_benchmark/ExKleisterCharity"
    "due_benchmark/ExPWC"
    "icdar2019/trackA_modern"
    "fintabnet/1k"
    "icdar2013/default"
)


declare -a big_datasets=(
    "rvlcdip/image_with_ocr_4k"
    "publaynet/4k"
    "doclaynet/1k"
    "doclaynet/4k"
    "pubtables1m/structure_4k"
    "fintabnet/4k"
)


if [[ "$1" == "small_datasets" ]]; then
    for dataset_entry in "${small_datasets[@]}"; do
        echo "Processing dataset: $name with config: $config and args: ${@:2}"
        uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2}
    done
elif [[ "$1" == "big_datasets" ]]; then
    for dataset_entry in "${big_datasets[@]}"; do
        echo "Processing dataset: $name with config: $config and args: ${@:2}"
        uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2}
    done
else
    dataset_entry="$1"
    echo "Processing dataset: $name with config: $config and args: ${@:2}"
    uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2}
fi