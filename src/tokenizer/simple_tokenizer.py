import json
import os
import re
import collections

class SimpleWordTokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        # Define special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Initialize empty mappings
        self.vocab = {}
        self.id_to_token = {}

    def _tokenize(self, text: str) -> list[str]:
        """
        Splits text into words and punctuation, ignoring whitespaces.
        """
        if self.lowercase:
            text = text.lower()
        # Find all words and punctuation characters
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts: list[str] | str, min_freq: int = 1):
        """
        Builds vocabulary from a list of texts or a single text, filtering out rare tokens.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Re-initialize vocabulary with special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        token_counts = collections.Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)
            
        # Add tokens to vocab if they meet the minimum frequency threshold
        idx = len(self.vocab)
        for token in sorted(token_counts.keys()):
            if token_counts[token] >= min_freq and token not in self.vocab:
                self.vocab[token] = idx
                self.id_to_token[idx] = token
                idx += 1

    def token_to_id(self, token: str) -> int:
        """
        Converts token to its corresponding ID. Falls back to <UNK>.
        """
        return self.vocab.get(token, self.vocab[self.unk_token])

    def id_to_token_fn(self, idx: int) -> str:
        """
        Converts ID to its corresponding token. Falls back to <UNK>.
        """
        return self.id_to_token.get(idx, self.unk_token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encodes raw text into a list of token IDs.
        """
        tokens = self._tokenize(text)
        ids = [self.token_to_id(t) for t in tokens]
        if add_special_tokens:
            ids = [self.token_to_id(self.bos_token)] + ids + [self.token_to_id(self.eos_token)]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token_fn(idx)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
            
        # Detokenize the list of words/punctuations
        detokenized = ""
        for i, token in enumerate(tokens):
            if i == 0:
                detokenized += token
            elif re.match(r"[^\w\s]", token):
                # Don't add space before punctuation
                detokenized += token
            else:
                detokenized += " " + token
        return detokenized

    def save_vocab(self, file_path: str):
        """
        Saves the vocabulary to a JSON file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path: str):
        """
        Loads the vocabulary from a JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vocabulary file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        # Rebuild reverse mapping (JSON loads keys as strings, convert to int)
        self.id_to_token = {int(idx): token for token, idx in self.vocab.items()}


class SimpleWordTokenizerV5(SimpleWordTokenizer):
    def __init__(self, lowercase: bool = True):
        super().__init__(lowercase=lowercase)
        self.eos_token = "<|eos|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.special_tokens = [
            self.pad_token, self.unk_token, self.bos_token, self.eos_token,
            self.system_token, self.user_token, self.assistant_token
        ]

    def _tokenize(self, text: str) -> list[str]:
        # Construct the regex dynamically from special tokens
        patterns = [re.escape(t.lower() if self.lowercase else t) for t in self.special_tokens]
        regex_pattern = "|".join(patterns) + r"|\w+|[^\w\s]"
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(regex_pattern, text, re.UNICODE)
        return tokens

