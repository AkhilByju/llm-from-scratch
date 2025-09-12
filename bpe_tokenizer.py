import json 
from collections import Counter
from pathlib import Path
from tqdm import tqdm 

class BPETokenizer:

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = []
    
    def get_stats(self, tokens):
        """Count frequency of adjacent symbol pairs in the tokenized text"""
        pairs = Counter()
        for word in tokens:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += 1
        return pairs
    
    def merge_pair(self, pair, tokens):
        """Merge the most frequent pair into a new symbol"""
        bigram = ''.join(pair)
        new_tokens = []
        for word in tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_tokens.append(new_word)
        return new_tokens
    
    def train(self, text, print_every=10, sample_size=None):
        """Train BPE merges from text"""
        if sample_size is not None:
            words = text.split(" ")[:sample_size]
            text = " ".join(words)

        tokens = [list(word) for word in text.split(" ")]
        for t in tokens:
            t.append(" ")   # re-add the space at the end of each word
        tokens[-1].pop()     # remove trailing space from last word

        # start vocab with characters + space + <unk>
        vocab = set(ch for word in tokens for ch in word)
        vocab.add(" ")
        vocab.add("<unk>")

        # tqdm progress bar
        for i in tqdm(range(self.vocab_size), desc="Training BPE merges"):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            tokens = self.merge_pair(best, tokens)
            new_symbol = ''.join(best)
            vocab.add(new_symbol)
            self.merges.append(best)

        # assign IDs
        self.vocab = {tok: i for i, tok in enumerate(sorted(vocab))}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, text, show_progress=False):
        """Efficient encode using BPE merges (linear-time per word)"""
        output = []
        words = text.split(" ")

        # optionally wrap with tqdm
        iterator = tqdm(words, desc="Encoding text") if show_progress else words

        for word in iterator:
            tokens = list(word)
            tokens.append(" ")  

            # Apply merges greedily
            merge_applied = True
            while merge_applied:
                merge_applied = False
                for a, b in self.merges:
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == a and tokens[i+1] == b:
                            tokens[i:i+2] = ["".join([a, b])]
                            merge_applied = True
                        else:
                            i += 1

            # Convert to IDs
            for tok in tokens:
                if tok in self.vocab:
                    output.append(self.vocab[tok])
                else:
                    output.append(self.vocab["<unk>"])

        return output


    
    def decode(self, ids):
        return "".join(self.inv_vocab.get(i, "<unk>") for i in ids)
    
    def save(self, path):
        """Save vocab + merges to JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "vocab": self.vocab,
                "merges": self.merges
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """Load vocab + merges from JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.inv_vocab = {int(i): tok for tok, i in self.vocab.items()} \
            if isinstance(list(self.vocab.values())[0], str) else {i: tok for tok, i in self.vocab.items()}
        self.merges = data["merges"]

"""
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

bpe = BPETokenizer(vocab_size=1000)
bpe.train(text, print_every=10, sample_size=200000)
bpe.save("bpe_vocab.json")

"""