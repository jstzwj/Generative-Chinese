
from tokenizers import CharBPETokenizer
import json
import tqdm

if __name__ == "__main__":
    # Initialize a tokenizer
    tokenizer = CharBPETokenizer()

    # Then train it!
    tokenizer.train([ "data\\train.txt" ])

    # Now, let's use it:
    encoded = tokenizer.encode("I can feel the magic, can you?")

    # And finally save it somewhere
    tokenizer.save("./", "bpe.tokenizer.json")