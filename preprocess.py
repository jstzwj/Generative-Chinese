import tqdm
import json
if __name__ == "__main__":
    dataset = []
    with open('./data/dataset.jsonl', 'r', encoding='utf-8') as f:
        for each_line in f:
            obj = json.loads(each_line)
            dataset.append(obj['content'])

    with open('./data/train.txt', 'w', encoding='utf-8') as f:
        for each_data in dataset:
            f.write(each_data + '\n')