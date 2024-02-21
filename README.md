# JointNER

Source code for COLING-2024's paper [Improving Chinese Named Entity Recognition with Multi-grained Words and Part-of-Speech Tags via Joint Modeling]().

## Requirements:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Train
**example**
```sh
python -m parser.cmd train -b \
    --mbr \
    --device 1 \
    --path exp/test.model \
    --encoder bert \
    --train data/train.tree \
    --dev data/dev.tree \
    --test data/test.tree \
    --conf config/ptb.crf.con.roberta.ini \ 
```

## Test
**example**
```sh
python -m parser.cmd evaluate \
    --device 1 \
    --path exp/test.model \
    --data data/test.tree \
```
