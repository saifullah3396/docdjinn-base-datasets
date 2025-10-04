# DocGenie Project Guide
## Setup
Install UV Astral for dependencies
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Prepare Datasets
Prepare the datasets in the datasets directory. 

### Classification Datasets
Prepare the classification datasets. This will automatically install all dependencies.
```
bash examples/integration/prepare_classification.sh ../datasets/
```

### Entity Labeling Datasets
Prepare the classification datasets. This will automatically install all dependencies and
```
bash examples/integration/prepare_entity_labeling.sh ../datasets/
```

### Extractive QA Datasets
Prepare the classification datasets. This will automatically install all dependencies and
```
bash examples/integration/prepare_extractive_qa.sh ../datasets/
```