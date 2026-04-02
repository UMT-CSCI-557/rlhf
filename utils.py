"""
Utility functions for the RLHF lab.

Data loading, text cleaning, vocabulary construction, and bigram count matrix
building — extracted from bigram_model.py and rlhf.py.
"""

import re
from collections import Counter

import polars as pl
import torch

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

# ---------------------------------------------------------------------------
# Text cleaning regexes
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_MEDIA_TAG_RE = re.compile(r"\[(?:image|video|photo|gif)\]", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_tweets(path: str) -> list[str]:
    """Read the CSV with polars and return a list of tweet texts."""
    df = pl.read_csv(path)
    texts = df.filter(pl.col("text").is_not_null())["text"].to_list()
    return texts


def clean_text(text: str) -> str:
    """Strip URLs, hashtags, media tags, emojis; lowercase; keep only letters + spaces."""
    text = _URL_RE.sub("", text)
    text = _HASHTAG_RE.sub("", text)
    text = _MEDIA_TAG_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(tokenized_tweets: list[list[str]], n: int) -> dict[str, int]:
    """Return a word -> index mapping for the top-n words plus special tokens."""
    counts = Counter()
    for tokens in tokenized_tweets:
        counts.update(tokens)
    most_common = [w for w, _ in counts.most_common(n)]
    vocab = {UNK: 0, BOS: 1, EOS: 2}
    for i, w in enumerate(most_common, start=len(vocab)):
        vocab[w] = i
    return vocab


# ---------------------------------------------------------------------------
# Bigram count matrix
# ---------------------------------------------------------------------------

def build_count_matrix(
    tokenized_tweets: list[list[str]],
    vocab: dict[str, int],
) -> dict[str, Counter]:
    """
    Build raw bigram counts as a nested dict: counts[w_prev][w_next] = int.
    """
    counts: dict[str, Counter] = {w: Counter() for w in vocab}
    for tokens in tokenized_tweets:
        if not tokens:
            continue
        mapped = [t if t in vocab else UNK for t in tokens]
        prev = BOS
        for tok in mapped:
            counts[prev][tok] += 1
            prev = tok
        counts[prev][EOS] += 1
    return counts


def counts_to_logprobs(counts: dict[str, Counter], alpha: float = 1e-3):
    """
    Convert a bigram count dict into a (V, V) log-probability tensor.

    Parameters
    ----------
    counts : dict from build_count_matrix
    alpha  : Laplace smoothing constant

    Returns
    -------
    logits : (V, V) float tensor of log P(next | prev)
    key2idx : dict mapping token strings to row/column indices
    idx2key : dict mapping indices back to token strings
    """
    V = len(counts)
    key2idx = {k: i for i, k in enumerate(counts.keys())}
    idx2key = {i: k for i, k in enumerate(counts.keys())}

    A = torch.ones((V, V), dtype=torch.float32) * alpha
    for k1, row in counts.items():
        for k2, col in row.items():
            A[key2idx[k1], key2idx[k2]] += col
    A /= A.sum(dim=1, keepdim=True)

    # EOS is an absorbing state
    eos_id = key2idx[EOS]
    A[eos_id] = 0.0
    A[eos_id, eos_id] = 1.0

    return torch.log(A), key2idx, idx2key
