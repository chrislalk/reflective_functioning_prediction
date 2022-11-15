from typing import List, Dict, Literal, TextIO
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def keep_token(token: str) -> bool:
    """
    Returns a boolean indicating whether a token should be kept
    """
    assert " " not in token
    assert "\t" not in token
    assert "\n" not in token
    assert "\r" not in token
    if len(token.strip()) == 0:
        return False
    return True


def tokenize(segment: str) -> List[str]:
    segment = segment.replace("\t", " ").strip()
    tokens = [_RE_COMBINE_WHITESPACE.sub(" ", token).strip() for token in segment.split(" ")]
    return [token for token in tokens if keep_token(token)]


def build_vocabulary(segments: List[str]) -> Dict[str, int]:
    dictionary = dict()
    for segment in segments:
        for line in segment:
            tokens = tokenize(line)
            for token in tokens:
                dictionary[token] = dictionary.get(token, 0) + 1
    return dictionary


def write_vocabulary_by_frequency(vocabulary: Dict[str, int], handle:TextIO,
                                  sorting: str = Literal["frequency", "lexicographic"]):
    if sorting == "frequency":
        vocab_sorted = sorted(vocabulary.items(), key=lambda x: -x[1])
    elif sorting == "lexicographic":
        vocab_sorted = sorted(vocabulary.items(), key=lambda x: x[0])
    for token, frequency in vocab_sorted:
        handle.write(f"{frequency}:\t{token}\n")
