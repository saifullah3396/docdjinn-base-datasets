from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

from atria_datasets.document_classification.rvlcdip import *  # noqa
from atria_datasets.document_classification.tobacco3482 import *  # noqa
from atria_datasets.image_classification.cifar10 import *  # noqa
from atria_datasets.image_classification.cifar10_huggingface import *  # noqa
from atria_datasets.image_classification.mnist import *  # noqa
from atria_datasets.language_modeling.due import *  # noqa
from atria_datasets.layout_analysis.doclaynet import *  # noqa
from atria_datasets.layout_analysis.icdar2019 import *  # noqa
from atria_datasets.layout_analysis.publaynet import *  # noqa
from atria_datasets.pipelines.atria_data_pipeline import *  # noqa
from atria_datasets.ser.cord import *  # noqa
from atria_datasets.ser.docbank import *  # noqa
from atria_datasets.ser.docile import *  # noqa
from atria_datasets.ser.funsd import *  # noqa
from atria_datasets.ser.sroie import *  # noqa
from atria_datasets.ser.wild_receipts import *  # noqa
from atria_datasets.table_extraction.fintabnet import *  # noqa
from atria_datasets.table_extraction.icdar2013 import *  # noqa
from atria_datasets.table_extraction.pubtables1m import *  # noqa
from atria_datasets.vqa.docvqa import *  # noqa
from atria_datasets.vqa.due import *  # noqa

if __name__ == "__main__":
    config_path = Path(__file__).parent / "conf"
    write_registry_to_yaml(
        config_path, types=["dataset", "data_pipeline", "batch_sampler"]
    )
