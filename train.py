import torch
import os
import json
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from itertools import islice, takewhile, repeat

from tokenizers import CharBPETokenizer, Tokenizer
from dataset import TextDataset
from model_gpt2 import GPT2LMHeadModel

def split_every(n, iterable):
    """
    Slice an iterable into chunks of n elements
    :type n: int
    :type iterable: Iterable
    :rtype: Iterator
    """
    iterator = iter(iterable)
    return takewhile(bool, (list(islice(iterator, n)) for _ in repeat(None)))


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    lines = []
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        for each_line in f:
            line = json.loads(each_line)
            line = line.replace('\n', ' [SEP] ') # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
            lines.append(line)
    
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    all_len = len(lines)
    piece_len = all_len // num_pieces
    for sublines in tqdm(iterable=list(split_every(piece_len, lines))):
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.bos_token)  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.eos_token)  # 文章之间添加CLS表示文章结束

        with open('tokenized_train.npy', 'wb') as f:
            np.savez(f, np.array(full_line))

    print('finish')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='模型训练batch大小')
    
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    batch_size = args.batch_size

    tokenizer = CharBPETokenizer("./vocab/bpe.tokenizer.json-vocab.json", './vocab/bpe.tokenizer.json-merges.txt')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # prepare dataset
    dataset = TextDataset(tokenizer, raw_data_path, 512)

    # model
    with open('./config/model_config.json', 'r', encoding='utf-8') as f:
        text = f.read()
        config = json.loads(text)
    model = GPT2LMHeadModel(config)
    model.train()
    model.to(device)

    # 打印参数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, sample in enumerate(dataloader):
        model.forward(sample)

if __name__ == "__main__":
    main()