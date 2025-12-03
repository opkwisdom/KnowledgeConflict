import re
import string
from typing import List

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

def check_answer(a_pred: str, a_true: List[str]) -> bool:
    """Simple grading function to compare predicted and true answers."""
    return any(normalize_answer(ans) in normalize_answer(a_pred) for ans in a_true)
