from typing import List, Dict, Literal, TextIO
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
REPLACE_TO_DOTS = [
    # since we do not know how many words are missing:
    re.compile(r"\(\s*unverständlich\s*\)"),
]


def keep_token(token: str) -> bool:
    """
    Returns a boolean indicating whether a token should be kept
    """
    assert " " not in token
    assert "\t" not in token
    assert "\n" not in token
    assert "\r" not in token
    token = token.strip()
    if len(token) == 0:
        return False
    return True


def strip_brackets(segment: str, start_char: str="(", end_char: str=")") -> str:
    """
    Remove brackets and all content within. Checks for every opening bracket if it is closed.
    :param segment: The text
    :param start_char: Which character opens the bracket?
    :param end_char: Which character closes the bracket?
    :return: The stripped text
    """
    assert start_char != end_char
    open_brackets = 0
    non_bracket_text = ""
    # inefficient implementation, but easy to understand
    for char in segment:
        if char == start_char:
            open_brackets += 1
        elif char == end_char:
            open_brackets -= 1
            if open_brackets < 0:
                raise ValueError("Found closing bracket without opening bracket")
        elif open_brackets == 0:
            non_bracket_text += char
    if open_brackets > 0:
        raise ValueError("Opening bracket was never closed")
    return non_bracket_text


def tokenize(segment: str) -> List[str]:
    segment = segment.replace("\t", " ").strip()
    # combine multiple whitespaces
    segment = _RE_COMBINE_WHITESPACE.sub(" ", segment)
    # replace some text with a dot
    for regexp in REPLACE_TO_DOTS:
        segment = regexp.sub(". ", segment)
    # strip brackets
    for start_char, end_char in [["(", ")"],
                                 ["[", "]"],
                                 ["{", "}"]]:
        segment = strip_brackets(segment, start_char=start_char, end_char=end_char)
    tokens = [token.strip() for token in segment.split(" ")]
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
