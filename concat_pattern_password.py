import argparse
import os

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

def create_output_file(base_output_path, file_count):
    return open(f"{base_output_path}_{file_count}.txt", 'w', encoding='utf-8', errors='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of training dataset after split", type=str, required=True)
    parser.add_argument("--output_path", help="base path of output datasets (ready for training)", type=str, required=True)
    args = parser.parse_args()

    input_dataset = args.dataset_path
    base_output_path = args.output_path
    f_in = open(input_dataset, 'r', encoding='utf-8', errors='ignore')

    lines = f_in.readlines()
    file_count = 0
    f_out = create_output_file(base_output_path, file_count)
    max_file_size = 100 * 1024 * 1024  # 100 MB in bytes
    current_file_size = 0

    for line in lines:
        password = line.rstrip()
        prompt = ' '.join(get_pattern(password))
        new_line = prompt + ' <SEP> ' + ' '.join(list(password)) + '\n'
        f_out.write(new_line)
        current_file_size += len(new_line.encode('utf-8'))

        if current_file_size >= max_file_size:
            f_out.close()
            file_count += 1
            f_out = create_output_file(base_output_path, file_count)
            current_file_size = 0

    f_out.close()
