# This file aims to get patterns rate from training set (cleaned).
from transformers import (
    GPT2LMHeadModel
)
import torch
from tokenizer import CharTokenizer
from concat_pattern_password import get_pattern
import argparse
import matplotlib.pyplot as plt
import numpy as np

def compute_log_likelihood(model, input_ids):
    model.eval()
    # Compute the log likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)  # The labels parameter causes the model to return the loss
        log_likelihood = outputs.loss.item()  # The loss is the negative log likelihood of the sequence

    return log_likelihood

def process_file(input_file, output_file, tokenizer, model):
    pattern_log_likelihoods = {}

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f_out:
        for line in lines:
            pattern, input_text = line.strip().split('<SEP>')
            input_text = input_text.strip()
            pattern = pattern.strip()

            tokenizer_forgen_result = tokenizer.encode_forgen(line.strip())
            input_ids = tokenizer_forgen_result.view([1, -1])

            log_likelihood = compute_log_likelihood(model, input_ids)
            if pattern not in pattern_log_likelihoods:
                pattern_log_likelihoods[pattern] = []
            pattern_log_likelihoods[pattern].append(log_likelihood)

            output_line = f"{pattern} <SEP> {input_text} <SEP> {log_likelihood}\n"
            f_out.write(output_line)

    return pattern_log_likelihoods

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, required=True)
    parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json')
    parser.add_argument("--dataset_path", help="path of training dataset", type=str, required=True)
    parser.add_argument("--output_path", help="save path of pattern rate", type=str, default="./generate/loglikehood.txt")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_file = args.vocabfile_path
    input_file = args.dataset_path
    output_file = args.output_path

    print(f'Load tokenizer.')
    tokenizer = CharTokenizer(vocab_file=vocab_file,
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                              )
    tokenizer.padding_side = "left"

    model = GPT2LMHeadModel.from_pretrained(model_path)

    pattern_log_likelihoods = process_file(input_file, output_file, tokenizer, model)
    print(f'done')
