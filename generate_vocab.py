
from tokenizers import CharBPETokenizer
import json
import tqdm

if __name__ == "__main__":
    # Initialize a tokenizer
    tokenizer = CharBPETokenizer()

    # Then train it!
    tokenizer.train([
        "data\\train.txt",
        "D:/数据/wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw",
        "D:/数据/webtext2019zh/web_text_raw.txt"
        ],
        vocab_size=30000, min_frequency=2,
        special_tokens=['<UNK>', '<BOS>', '<EOS>', '<PAD>', '<CLS>', '<SEP>'])

    # Now, let's use it:
    encoded = tokenizer.encode("I can feel the magic, can you?")

    # And finally save it somewhere
    tokenizer.save("./", "bpe.tokenizer.json")