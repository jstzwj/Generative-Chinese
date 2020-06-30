from typing import List, Dict, Union, Iterable

class Tokenizer(object):
    def __init__(self):
        self.special_tokens = {}
        self.tokens = []

        self.token_to_id = {}
        self.id_to_token = {}

    @classmethod
    def from_vocab(path):
        tokenizer = Tokenizer()
        with open(path, 'r', encoding='utf-8') as f:
            for each_line in f:
                if each_line.endswith('\n'):
                    token = each_line[:-1]
                    tokenizer.add_token(token)

        return tokenizer

    def add_special_token(self, token:str, special_name:str):
        if token in self.token_to_id.keys():
            raise Exception('This token has been added to vocab')
        self.tokens.append(token)
        self.special_tokens[special_name] = token
        self.token_to_id[token] = len(self.tokens) - 1
        self.id_to_token[len(self.tokens) - 1] = token

        if special_name == 'bos_token':
            self.bos_token = token
        elif special_name == 'eos_token':
            self.eos_token = token
        elif special_name == 'unk_token':
            self.unk_token = token
        elif special_name == 'sep_token':
            self.sep_token = token
        elif special_name == 'pad_token':
            self.pad_token = token
        elif special_name == 'cls_token':
            self.cls_token = token
        elif special_name == 'mask_token':
            self.mask_token = token
    
    def add_special_tokens(self, tokens:Dict[str, str]):
        if 'bos_token' in tokens:
            self.add_special_token(tokens['bos_token'], 'bos_token')
        if 'eos_token' in tokens:
            self.add_special_token(tokens['eos_token'], 'eos_token')
        if 'unk_token' in tokens:
            self.add_special_token(tokens['unk_token'], 'unk_token')
        if 'sep_token' in tokens:
            self.add_special_token(tokens['sep_token'], 'sep_token')
        if 'pad_token' in tokens:
            self.add_special_token(tokens['pad_token'], 'pad_token')
        if 'cls_token' in tokens:
            self.add_special_token(tokens['cls_token'], 'cls_token')
        if 'mask_token' in tokens:
            self.add_special_token(tokens['mask_token'], 'mask_token')
        if 'additional_special_tokens' in tokens:
            self.add_special_token(tokens['additional_special_tokens'], 'additional_special_tokens')

    def add_token(self, token:str):
        if token in self.token_to_id.keys():
            raise Exception('This token has been added to vocab')
        self.tokens.append(token)
        self.token_to_id[token] = len(self.tokens) - 1
        self.id_to_token[len(self.tokens) - 1] = token
    
    def add_tokens(self, tokens:Iterable[str]):
        for each_token in tokens:
            self.add_token(each_token)

class CharTokenizer(Tokenizer):
    def __init__(self):
        super(CharTokenizer, self).__init__()

    @classmethod
    def from_vocab(cls, path):
        special_tokens = {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'unk_token': '[UNK]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'mask_token': '[MASK]',
        }
        tokenizer = CharTokenizer()
        tokenizer.add_special_tokens(special_tokens)
        with open(path, 'r', encoding='utf-8') as f:
            for each_line in f:
                if each_line.endswith('\n'):
                    token = each_line[:-1]
                    if token in special_tokens.values():
                        continue
                    tokenizer.add_token(token)
        return tokenizer

    def tokenize(self, text):
        return [c for c in text]

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for each_token in tokens:
            if each_token in self.token_to_id.keys():
                ids.append(self.token_to_id[each_token])
            else:
                ids.append(self.unk_token)
        return ids