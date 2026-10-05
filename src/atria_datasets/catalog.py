"""
Dataset Catalog

A plain lookup table mapping `"<dataset_name>/<config_name>"` to the dataset class
(as an import path, so heavy dataset modules are only imported when used) and the
config kwargs passed to its constructor.
"""

import importlib
from typing import Any

_DOCLAYNET_HF = {"hf_repo": "ds4sd/DocLayNet", "hf_config_name": "2022.08"}
_PUBLAYNET_HF = {"hf_repo": "jordanparker6/publaynet", "hf_config_name": "default"}

_RVLCDIP = "atria_datasets.document_classification.rvlcdip.RvlCdip"
_TOBACCO3482 = "atria_datasets.document_classification.tobacco3482.Tobacco3482"
_DOCLAYNET = "atria_datasets.layout_analysis.doclaynet.DocLayNet"
_ICDAR2019 = "atria_datasets.layout_analysis.icdar2019.Icdar2019"
_PUBLAYNET = "atria_datasets.layout_analysis.publaynet.PubLayNet"
_FINTABNET = "atria_datasets.table_extraction.fintabnet.FinTabNet"
_PUBTABLES1M = "atria_datasets.table_extraction.pubtables1m.PubTables1M"
_DUE_EXTRACTIVE_QA = "atria_datasets.vqa.due.DueBenchmarkExtractiveQA"

DATASETS: dict[str, tuple[str, dict[str, Any]]] = {
    # document classification
    "tobacco3482/image_with_ocr": (_TOBACCO3482, {"load_ocr": True}),
    "rvlcdip/image_with_ocr_4k": (
        _RVLCDIP,
        {"load_ocr": True, "max_train_samples": 4000, "max_validation_samples": 4000},
    ),
    # semantic entity recognition
    "cord/default": ("atria_datasets.ser.cord.CORD", {}),
    "funsd/default": ("atria_datasets.ser.funsd.FUNSD", {}),
    "sroie/default": ("atria_datasets.ser.sroie.SROIE", {}),
    "wild_receipts/default": ("atria_datasets.ser.wild_receipts.WildReceipts", {}),
    "docile/default": ("atria_datasets.ser.docile.Docile", {}),
    # extractive qa (due benchmark)
    "due_benchmark/ExDocVQA": (_DUE_EXTRACTIVE_QA, {}),
    "due_benchmark/ExDeepForm": (_DUE_EXTRACTIVE_QA, {}),
    "due_benchmark/ExTabFact": (_DUE_EXTRACTIVE_QA, {"ocr_engine": "tesseract"}),
    "due_benchmark/ExWikiTableQuestions": (_DUE_EXTRACTIVE_QA, {}),
    "due_benchmark/ExInfographicsVQA": (_DUE_EXTRACTIVE_QA, {}),
    "due_benchmark/ExKleisterCharity": (_DUE_EXTRACTIVE_QA, {}),
    "due_benchmark/ExPWC": (_DUE_EXTRACTIVE_QA, {"ocr_engine": "tesseract"}),
    # layout analysis
    "publaynet/4k": (
        _PUBLAYNET,
        {**_PUBLAYNET_HF, "max_train_samples": 4000},  # val set is same as test set
    ),
    "doclaynet/1k": (
        _DOCLAYNET,
        {**_DOCLAYNET_HF, "max_train_samples": 1000, "max_validation_samples": 1000},
    ),
    "doclaynet/4k": (
        _DOCLAYNET,
        {**_DOCLAYNET_HF, "max_train_samples": 4000, "max_validation_samples": 4000},
    ),
    "icdar2019/trackA_modern": (_ICDAR2019, {}),
    # table extraction
    "pubtables1m/structure_4k": (
        _PUBTABLES1M,
        {
            "task": "structure",
            "max_train_samples": 4000,
            "max_validation_samples": 4000,
        },
    ),
    "fintabnet/1k": (
        _FINTABNET,
        {
            "max_train_samples": 1000,
            "max_validation_samples": 1000,
            "max_test_samples": 1000,
        },
    ),
    "fintabnet/4k": (
        _FINTABNET,
        {
            "max_train_samples": 4000,
            "max_validation_samples": 4000,
            "max_test_samples": 4000,
        },
    ),
    "icdar2013/default": ("atria_datasets.table_extraction.icdar2013.ICDAR2013", {}),
}


def get_dataset(name: str, **overrides):
    """
    Instantiate a dataset from the catalog.

    Args:
        name: `"<dataset_name>/<config_name>"`; a bare `"<dataset_name>"` means the `default` config
        **overrides: Config kwargs overriding the catalog entry

    Returns:
        The (unbuilt) dataset instance
    """
    if "/" not in name:
        name = f"{name}/default"
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets:\n" + "\n".join(DATASETS)
        )
    target, kwargs = DATASETS[name]
    dataset_name, config_name = name.split("/", 1)
    module_path, cls_name = target.rsplit(".", 1)
    dataset_cls = getattr(importlib.import_module(module_path), cls_name)
    return dataset_cls(
        dataset_name=dataset_name, config_name=config_name, **{**kwargs, **overrides}
    )
