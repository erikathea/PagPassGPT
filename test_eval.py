from transformers import (
    GPT2LMHeadModel
)
import time
import datetime
import threading
import torch
from tokenizer import CharTokenizer
import argparse
from itertools import combinations_with_replacement, permutations, product
import random
import re
import logging
import os
import string
from zxcvbn import zxcvbn
from Levenshtein import distance, ratio

MAX_LEN = 32
logging.basicConfig(filename='./test-pass.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# We are generating password variants that comply to the minimum common modern password policy:
# 1 uppercase letter, 1 lowercase letter, 1 digit, 1 symbol pattern (u1-l1-d1-s1)

# PagPassGPT simplify tokenization by being case insensitive.
# So we have to randomly upcase a letter from the generated password variant
# and ensure that the password variant pattern is L-S-N
# with minimum length set to 8 

def generate_password_pattern(permutation, pw_len):
    parts = permutation.split()
    counts = {part[0]: int(part[1:]) for part in parts}
    max_len = max(8, pw_len)
    
    for letter in 'LNS':
        if letter not in counts:
            counts[letter] = 0
    
    total = sum(counts.values())
    while sum(counts.values()) != max_len or min(counts.values()) == 0:
        if sum(counts.values()) < max_len:
            # Add to missing or smallest count
            candidates = [l for l in 'LNS' if counts[l] == min(counts.values())]
            letter = random.choice(candidates)
            counts[letter] += 1
        else:
            # Subtract from largest count
            candidates = [l for l in 'LNS' if counts[l] == max(counts.values()) and counts[l] > 1]
            if candidates:
                letter = random.choice(candidates)
                counts[letter] -= 1
            else:
                # If we can't subtract, redistribute
                for l in 'LNS':
                    if counts[l] > 1:
                        counts[l] -= 1
                        break
    
    parts = [f"{letter}{count}" for letter, count in counts.items() if count > 0]
    all_permutations = list(permutations(parts))
    
    permutation_strings = [' '.join(perm) for perm in all_permutations]
    return permutation_strings

def get_pattern(password: str):
    result = []
    current_type = None
    current_length = 0

    for char in password:
        if char.isalpha():
            if current_type == 'L':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'L'
                current_length = 1
        elif char.isdigit():
            if current_type == 'N':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'N'
                current_length = 1
        else:
            if current_type == 'S':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'S'
                current_length = 1

    if current_type:
        result.append(current_type + str(current_length))
    return result

def remove_random_character(s):
    if not s:
        return s
    index = random.randint(0, len(s) - 1)  # Choose a random index
    return s[:index] + s[index + 1:] 

def ensure_case_diversity(password):
    if has_both_cases(password):
        return password
    
    password_chars = list(password)
    
    if not has_uppercase(password):
        add_uppercase(password_chars)
    
    if not has_lowercase(password):
        add_lowercase(password_chars)
    
    return ''.join(password_chars)

def has_both_cases(string):
    return has_uppercase(string) and has_lowercase(string)

def has_uppercase(string):
    return any(char.isupper() for char in string)

def has_lowercase(string):
    return any(char.islower() for char in string)

def add_uppercase(chars):
    change_random_char(chars, str.islower, str.upper)

def add_lowercase(chars):
    change_random_char(chars, str.isupper, str.lower)

def change_random_char(chars, condition, transform):
    allowed_chars = string.ascii_letters
    eligible_indices = [i for i, char in enumerate(chars) if condition(char)]
    if eligible_indices:
        index = random.choice(eligible_indices)
        transformed_char = transform(chars[index])
        # Ensure the transformed character is in the allowed set
        while transformed_char not in allowed_chars:
            transformed_char = transform(random.choice(list(allowed_chars)))
        chars[index] = transformed_char

def valid_format(password):
    pattern = r'^(?=.*[!@#$%^&*()_+\-=[\]{}|;:\'",.<>/?`~])(?=.*\d).+$'
    return bool(re.match(pattern, password))

def compute_log_likelihood(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)  # The labels parameter causes the model to return the loss
        log_likelihood = outputs.loss.item()  # The loss is the negative log likelihood of the sequence
    return log_likelihood

def get_log_filename(input_password):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_password = ''.join(c if c.isalnum() else '_' for c in input_password[:10])
    return f"password_variant_{safe_password}_{timestamp}.log"

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, default='./model/checkpoint-384000')
    parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json')
    parser.add_argument("--input_file", help="file containing passwords to be used for creating variants", type=str, required=True)
    parser.add_argument("--generate_num", help="query budget per pattern", default=10, type=int)
    parser.add_argument("--compute_loglikelihood", help="compute log likelihood of the passwords", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_file = args.vocabfile_path

    tokenizer = CharTokenizer(vocab_file=vocab_file, 
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                              )
    allowed_chars = set(tokenizer.encoder.keys())
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    with open(args.input_file, 'r') as file:
        lines = file.read().splitlines()

    logger.info(f'\torig_pw\torig_pattern\torig_zxcvbn_score\torig_zxcvbn_guesses\torig_log_likelihood\tpw_variant\tpattern\tpw_zxcvbn_score\tpw_zxcvbn_guesses\tpw_log_likelihood\tedit_distance\tsimilarity_ratio')
    for line in lines:
        start_time = time.time()
        inputs = set()
        ip, pw = line.strip().split('<SEP>')
        ip = ip.strip()
        pw = ''.join(pw.strip().split(' '))
        orig_pw = ''.join(pw.strip().split(' '))
        orig_zxcvbn = zxcvbn(orig_pw)
        orig_pattern = ''.join(ip.strip().split(' '))

        forgen_result = tokenizer.encode_forgen(line)
        input_id = forgen_result.view([1, -1])
        orig_loglikelihood = compute_log_likelihood(model, input_id)

        fps = generate_password_pattern(ip, len(pw))
        for fp in fps:
            inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)) )

        # simulate Das-R rule: Delete Character
        pw = remove_random_character(pw)
        ip = ' '.join(get_pattern(pw))
        fps = generate_password_pattern(ip, len(pw))
        for fp in fps:
            inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))

        pw = remove_random_character(pw)
        ip = ' '.join(get_pattern(pw))
        fps = generate_password_pattern(ip, len(pw))
        for fp in fps:
            inputs.add(fp + ' <SEP> ' + ' '.join(list(pw)))

        print(f'{inputs}')
        tokenizer_forgen_results = [tokenizer.encode_forgen(input_text) for input_text in inputs]
        passwords = set()
    
        for tokenizer_forgen_result in tokenizer_forgen_results:
            input_ids=tokenizer_forgen_result.view([1, -1])
            outputs = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id,
                max_length=MAX_LEN,
                do_sample=True,
                num_return_sequences=args.generate_num
            )
            decoded_outputs = tokenizer.batch_decode(outputs)
            for output in decoded_outputs:
                pattern, pw_variant = output.split(' ', 1)
                try:
                    if valid_format(pw_variant) and len(pw_variant) >= 8:
                        password = ensure_case_diversity(pw_variant)
                        if all(char in allowed_chars for char in password) and password not in passwords:
                            passwords.add(password)

                            pattern = get_pattern(password)
                            input_pw = ' '.join(pattern) + ' <SEP> ' + ' '.join(list(password))
                            forgen_result = tokenizer.encode_forgen(input_pw)
                            input_id = forgen_result.view([1, -1])
                            log_likelihood = compute_log_likelihood(model, input_id)

                            pw_zxcvbn = zxcvbn(password)
                            edit_distance = distance(orig_pw, password)
                            similarity_ratio = ratio(orig_pw, password)
                            logger.info(f'\t{orig_pw}\t{orig_pattern}\t{orig_zxcvbn['score']}\t{orig_zxcvbn['guesses']}\t{orig_loglikelihood}\t{password}\t{''.join(pattern)}\t{pw_zxcvbn['score']}\t{pw_zxcvbn['guesses']}\t{log_likelihood}\t{edit_distance}\t{similarity_ratio}')
                except Exception as e:
                    print(f"Error encoding input '{output}': {str(e)}")
                    continue
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        print(f"password variant generation done for {orig_pw}, time: {total_time:.6f} ms ")
