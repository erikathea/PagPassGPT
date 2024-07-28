# PagPassGPT - Password Variant Generator

To generate password variants from an input:

    python generate_pw_variant.py --input_password="password" --generate_num="10" --compute_loglikelihood

The script is expecting `./tokenizer/vocab.json'` and `./model/checkpoint-384000`

Remove `--compute_loglikelihood` to skip computation and speed-up runtime.

# PagPassGPT
PagPassGPT: Pattern Guided Password Guessing via Generative Pretrained Transformer

Paper: https://arxiv.org/abs/2404.04886v1

## 1 Environment

```shell
pip install torch transformers datasets
pip install transformers[torch]
```

## 2 How to use

### 2.1 Train a PagPassGPT

Firstly, you should have a dataset of passwords, like `rockyou` or other datasets. And you should make sure the dataset contains only passwords.

Next, run the script to preprocess datasets.
```shell
sh ./scripts/preprocess.sh
```

Last, run the script to train.
```shell
sh ./scripts/train.sh
```

### 2.2 Generate passwords

```shell
sh ./scripts/generate.sh
```

In this shell, you can choose to use DC-GEN or not by changing just one line.

