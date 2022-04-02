# TPORE

Source code for **TPORE**, described by the paper: [End-to-End Open Relation Extraction in Chinese Field](https://arxiv.org/pdf/2010.12812.pdf).

In this paper, we selected two Chinese Open Relation Extraction Dataset for experiments.

* **COER** dataset comes from the paper [Chinese Open Relation Extraction and Knowledge Base Establishment](https://hong.xmu.edu.cn/__local/7/11/EF/278F61A2A2874569C391BBD78A8_5A45CBFF_227EB3.pdf?e=.pdf). We can find its dataset from [https://github.com/TJUNLP/COER](https://github.com/TJUNLP/COER).

* **SpanSAOKE** dataset comes from the paper [Multi-Grained Dependency Graph Neural Network for Chinese Open Information Extraction](https://link.springer.com/chapter/10.1007/978-3-030-75768-7_13). We can find its dataset from [https://github.com/Lvzhh/MGD-GNN](https://github.com/Lvzhh/MGD-GNN).

Also, we need to download the pretrained language model: [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main).

## Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```

## Quick Start

First, the original input file for our model needs to be `.jsonl` and each data in the file should be in following format:
```python
# Assume that we have predefined FRE `BirthDate` but not predefined `BirthPlace`.
{
    "text": "1980年姚明出生于上海",
    "relation_list": [
        # 开放关系中predicate来自于文本，需要有明确的predicate_span
        {
            "subject": "姚明",
            "predicate": "出生于",
            "object": "上海",
            "subject_span": [5, 7],
            "predicate_span": [7, 10],
            "object_span": [10, 12]
        },
        {
            "subject": "姚明",
            "predicate": "出生于",
            "object": "1980年",
            "subject_span": [5, 7],
            "predicate_span": [7, 10],
            "object_span": [0, 5]
        },
        # 限定关系中predicate不需要来自于文本，predicate_span此keykey为空或者[-1, -1]
        {
            "subject": "姚明",
            "predicate": "BirthDate",
            "object": "1980年",
            "subject_span": [5, 7],
            "predicate_span": [-1, -1],
            "object_span": [0, 5]
        },
    ],
    "entity_list": [
        {
            "text": "1980年",
            "entity_span": [0, 5],
            "type": "Time"
        },
        {
            "text": "姚明",
            "entity_span": [5, 7],
            "type": "Person"
        },
        {
            "text": "上海",
            "entity_span": [10, 12],
            "type": "Location"
        },

    ]
}
```

Then we execute the following command to convert the data into the format required by the model:

```bash
python preprocess_data.py
```

The converted data format is as follows:

```python
{
    "text": "1980年姚明出生于上海", 
    "labels": [
        # entity recognition
        [1, 5, "ENT_object"], 
        [1, 5, "ENT_Time"], 
        [6, 7, "ENT_subject"], 
        [6, 7, "ENT_Person"], 
        [8, 10, "ENT_predicate"]
        [11, 12, "ENT_object"],
        [11, 12, "ENT_Location"],

        # ORE labels
        [6, 8, "ORE_SH2PH"], 
        [7, 10, "ORE_ST2PT"], 
        [1, 8, "ORE_OH2PH"], 
        [5, 10, "ORE_OT2PT"], 
        [6, 1, "ORE_SH2OH"], 
        [7, 5, "ORE_ST2OT"]
        [11, 8, "ORE_OH2PH"], 
        [12, 10, "ORE_OT2PT"], 
        [6, 11, "ORE_SH2OH"], 
        [7, 12, "ORE_ST2OT"]
        
        # FRE labels
        [6, 1, "FRE_SH2OH_BirthDate"],
        [7, 5, "FRE_SH2OH_BirthDate"],
    ]
}

```

The converted file should be under the path:  `TPORE/data/data4model/`.

For different datasets, we should modify the labels in file `TPORE/label_config.json`.

```json
{
    "subject_type": [
        "subject"
    ],
    "predicate_type": [
        "predicate"
    ],
    "object_type": [
        "object",
        "place",
        "time",
        "qualifier"
    ],
    "ner": [],
    "fr": [
        "IN",
        "ISA",
        "DESC",
        "BIRTH",
        "DEATH"
    ]
}
```


Finally, we execute *run_saoke.sh* or the following command to train:
```bash
python3 -u main.py \
        --task_name [Your task name] \
        --train_file [data/data4model/xx.h3.train.jsonl] \
        --train_file [data/data4model/xx.h3.dev.jsonl] \
        --do_train \
        --batch_size 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        --max_seq_len 100 \
        --eval_epoch 1 \
        --output_dir [The folder where the training model is saved] \
        --pretrained_model_path [bert-base-chinese dirname | Your Output_dir] \
        --vocab_path [bert-base-chinese dirname]
```

And we execute the following command to test:
```bash
python3 -u main.py \
        --task_name [Your task name] \
        --test_file [data/data4model/xx.h3.test.jsonl] \
        --do_predict \
        --batch_size 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        --max_seq_len 100 \
        --eval_epoch 1 \
        --output_dir [The folder where the training model is saved] \
        --pretrained_model_path [Your Output_dir] \
        --vocab_path [bert-base-chinese dirname]
```


## Questions
If you have any questions about the code, please file an issue or contact us.