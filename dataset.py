import os
import time
import pickle
import logging
from filelock import FileLock

import torch
import numpy as np
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Tokenizer, file_path: str, block_size: int, overwrite_cache=False):
        super(TextDataset, self).__init__()
        self.path = file_path
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.data = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.data = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.encode(text)

                for i in range(0, len(tokenized_text.ids) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.data.append(
                        tokenized_text.ids[i : i + block_size]
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __getitem__(self, index:int) -> torch.Tensor:
        return torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)