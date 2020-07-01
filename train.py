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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='模型训练batch大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    print('start training')
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model.forward(input_ids=sample, labels=sample)
            loss, logits = outputs[:2]

            # backward
            loss.backward()
            optimizer.step()

            # print
            print(f'now time: {datetime.now().strftime("%H:%M")}. \
                Step {i} of epoch {epoch}, loss {loss.item()}')

        # save model
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_epoch_dir = os.path.join(output_dir, f'epoch_{str(epoch)}')
        if not os.path.exists(output_epoch_dir):
            os.mkdir(output_epoch_dir)
        torch.save(model.state_dict(), os.path.join(output_epoch_dir, 'model.pth'))

if __name__ == "__main__":
    main()