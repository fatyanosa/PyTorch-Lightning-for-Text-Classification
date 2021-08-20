# PyTorch-Lightning for Text Classification [![DOI](https://zenodo.org/badge/398056958.svg)](https://zenodo.org/badge/latestdoi/398056958)
Rank #55 in [GLUE Benchmark Leaderboard](https://gluebenchmark.com/leaderboard) using distilbert-base-uncased with manually tuned hyperparameters.

## TRAINING

### 1 Load from local files
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --train_data data/suggestion_mining/train.csv --dev_data data/suggestion_mining/dev.csv --metric f1 --monitor f1 --text_fields 'sentence' --class_names '0 1'
```
### 2 Load from datasets library

* CoLA
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name cola  --monitor matthews_correlation
```

* SST-2
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name sst2 --monitor accuracy
```

* MRPC
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name mrpc  --monitor combined_score
```

* STS-B
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name stsb  --monitor mse --metric_mode min
```

* QQP
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name qqp  --monitor combined_score
```

* MNLI
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name mnli  --monitor combined_score
```

* QNLI
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name qnli --monitor accuracy
```

* RTE
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name rte --monitor accuracy
```

* WNLI
``` bash
python training.py --gpus 1 --model_name_or_path distilbert-base-uncased --task_name wnli --monitor accuracy
```


## VALIDATION

### 1 Load from local files
``` bash
python testing.py --test_data data/suggestion_mining/dev.csv --model experiments/suggestion_mining --output_file data/suggestion_mining/dev.tsv --label_form index
```

### 2 Load from datasets library

* CoLA
``` bash
python testing.py --task_name cola --model experiments/distilbert-base-uncased/CoLA --output_file data/submission/CoLA.tsv --label_form index --test_type validation
```

* SST-2
``` bash
python testing.py --task_name sst2 --model experiments/distilbert-base-uncased/SST-2 --output_file data/submission/SST-2.tsv --label_form index --test_type validation
```

* MRPC
``` bash
python testing.py --task_name mrpc --model experiments/distilbert-base-uncased/MRPC --output_file data/submission/MRPC.tsv --label_form index --test_type validation
```

* STS-B
``` bash
python testing.py --task_name stsb --model experiments/distilbert-base-uncased/STS-B --output_file data/submission/STS-B.tsv --test_type validation
```

* QQP
``` bash
python testing.py --task_name qqp --model experiments/distilbert-base-uncased/QQP --output_file data/submission/QQP.tsv --label_form index --test_type validation
```

* MNLI-matched
``` bash
python testing.py --task_name mnli_matched --model experiments/distilbert-base-uncased/MNLI --output_file data/submission/MNLI-m.tsv --label_form names --test_type validation
```

* MNLI-matched
``` bash
python testing.py --task_name mnli_mismatched --model experiments/distilbert-base-uncased/MNLI --output_file data/submission/MNLI-mm.tsv --label_form names --test_type validation
```

* QNLI
``` bash
python testing.py --task_name qnli --model experiments/distilbert-base-uncased/QNLI --output_file data/submission/QNLI.tsv --label_form names --test_type validation
```

* RTE
``` bash
python testing.py --task_name rte --model experiments/distilbert-base-uncased/RTE --output_file data/submission/RTE.tsv --label_form names --test_type validation
```

* WNLI
``` bash
python testing.py --task_name wnli --model experiments/distilbert-base-uncased/WNLI --output_file data/submission/WNLI.tsv --label_form index --test_type validation
```


## TESTING

### 1 Load from local files
``` bash
python testing.py --test_data data/suggestion_mining/test.csv --model experiments/suggestion_mining --output_file data/suggestion_mining/test.tsv --label_form index
```

### 2 Load from datasets library

* CoLA
``` bash
python testing.py --task_name cola --model experiments/distilbert-base-uncased/CoLA --output_file data/submission/CoLA.tsv --label_form index
```

* SST-2
``` bash
python testing.py --task_name sst2 --model experiments/distilbert-base-uncased/SST-2 --output_file data/submission/SST-2.tsv --label_form index
```

* MRPC
``` bash
python testing.py --task_name mrpc --model experiments/distilbert-base-uncased/MRPC --output_file data/submission/MRPC.tsv --label_form index
```

* STS-B
``` bash
python testing.py --task_name stsb --model experiments/distilbert-base-uncased/STS-B --output_file data/submission/STS-B.tsv
```

* QQP
``` bash
python testing.py --task_name qqp --model experiments/distilbert-base-uncased/QQP --output_file data/submission/QQP.tsv --label_form index
```

* MNLI-matched
``` bash
python testing.py --task_name mnli_matched --model experiments/distilbert-base-uncased/MNLI --output_file data/submission/MNLI-m.tsv --label_form names
```

* MNLI-matched
``` bash
python testing.py --task_name mnli_mismatched --model experiments/distilbert-base-uncased/MNLI --output_file data/submission/MNLI-mm.tsv --label_form names
```

* QNLI
``` bash
python testing.py --task_name qnli --model experiments/distilbert-base-uncased/QNLI --output_file data/submission/QNLI.tsv --label_form names
```

* RTE
``` bash
python testing.py --task_name rte --model experiments/distilbert-base-uncased/RTE --output_file data/submission/RTE.tsv --label_form names
```

* WNLI
``` bash
python testing.py --task_name wnli --model experiments/distilbert-base-uncased/WNLI --output_file data/submission/WNLI.tsv --label_form index
```

* AX
``` bash
python testing.py --task_name ax --model experiments/distilbert-base-uncased/MNLI --output_file data/submission/AX.tsv --label_form names
```

# Cite As
Tirana Noor Fatyanosa. 2021. PyTorch-Lightning for Text Classification. Zenodo. https://doi.org/10.5281/zenodo.5224947
