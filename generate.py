
import os
import json

import torch
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_gpt2 import GPT2LMHeadModel
from tokenizers import CharBPETokenizer, Tokenizer

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.topk(logits, k=k, dim=-1)
        min_values = values[:, -1, None]

        logits[logits < min_values] = -1e10
        return logits

    if k == 0:
        return logits
    else:
        return _top_k()

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch = logits.shape[0]
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -1e10

    return logits

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token)
    
    def step(model, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--raw_data_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='模型推断batch大小')
    parser.add_argument('--model_path', default='./model/epoch_5/model.bin', type=str, required=False, help='模型保存位置')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    model_path = args.model_path

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # tokenizer
    tokenizer = CharBPETokenizer("./vocab/bpe.tokenizer.json-vocab.json", './vocab/bpe.tokenizer.json-merges.txt')

    # model
    with open('./config/model_config.json', 'r', encoding='utf-8') as f:
        text = f.read()
        config = json.loads(text)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

if __name__ == "__main__":
    main()