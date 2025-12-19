import re
import string
import regex
from collections import Counter
from typing import List
from dataclasses import dataclass


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))

def has_answer(a_pred: str, a_true: List[str], tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    a_pred = normalize_answer(a_pred)
    pred_tokens = tokenizer.tokenize(a_pred, uncased=True)

    for ans in a_true:
        ans = normalize_answer(ans)
        ans = tokenizer.tokenize(ans, uncased=True)
        for i in range(0, len(pred_tokens) - len(ans) + 1):
            if ans == pred_tokens[i: i + len(ans)]:
                return True
    return False

def em_for_interpretability(a_pred: str, a_true: List[str]) -> bool:
    """Check if a document contains an answer string EXACTLY."""
    return any([ans in a_pred for ans in a_true])