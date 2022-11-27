from typing import List, Dict, Literal, TextIO
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
REPLACE_TO_DOTS = [
    # since we do not know how many words are missing when transcribers could not understand something:
    re.compile(r"\(\s*unverständlich\s*\)"),
    re.compile(r"/"),
    re.compile(r"\?"),
    re.compile(r"\!"),
]
_RE_VALID_WORD = re.compile(r"^[äöüßÄÖÜa-zA-Z]*[0-9]*$")
_RE_HYPHENATED_WORD = re.compile(f"^[äöüßÄÖÜa-zA-Z]+(-[äöüßÄÖÜa-zA-Z]+)+$")


def keep_token(token: str) -> bool:
    """
    Returns a boolean indicating whether a token should be kept
    """
    assert " " not in token
    assert "\t" not in token
    assert "\n" not in token
    assert "\r" not in token
    token = token.strip()
    if len(token) == 0 or \
            token == "P:" or\
            token.endswith("-") or\
            token == "+" or\
            token == "-" or\
            token == '–':
        return False
    if _RE_VALID_WORD.match(token) is None and not token == ".":
        # word does not match rule
        # see if it is an anonymized name/place etc.
        if token.endswith("*") and _RE_VALID_WORD.match(token[:-1]) is not None:
            pass
        else:
            raise ValueError(f"Not a word: {token}")
    return True


def strip_brackets(segment: str, start_char: str = "(", end_char: str = ")") -> str:
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


def prepare_segment_for_tokenization(segment: str) -> str:
    """
    Apply various replacements, strip brackets
    """
    # transcribers use commas more like enumerations, so do not use them
    segment = segment.replace(",", ". ")
    # strip quotation marks and other special characters
    segment = segment.replace('"', "")
    segment = segment.replace('„', "")
    segment = segment.replace('“', "")
    segment = segment.replace('+', "")
    # combine multiple whitespaces
    segment = _RE_COMBINE_WHITESPACE.sub(" ", segment)
    segment = segment.strip()
    # replace some text with a dot
    for regexp in REPLACE_TO_DOTS:
        segment = regexp.sub(". ", segment)
    # strip brackets
    for start_char, end_char in [["(", ")"],
                                 ["[", "]"],
                                 ["{", "}"]]:
        segment = strip_brackets(segment, start_char=start_char, end_char=end_char)
    return segment


def split_hyphen_nouns(tokens: List[str]) -> List[str]:
    """
    Search for tokens containing composite nouns joined by a hyphen, break them into multiple tokens
    """
    split_tokens = []
    for token in tokens:
        if _RE_HYPHENATED_WORD.match(token) is not None:
            split_tokens += token.split("-")
        else:
            split_tokens += [token]
    return split_tokens


def tokenize(segment: str) -> List[str]:
    """
    Prepares segment and tokenizes. A dot will be preserved as a separate token.
    """
    segment = prepare_segment_for_tokenization(segment)
    tokens = []
    current_token = ""
    # iterate char-wise for simple algorithm
    for char in segment:
        if char == " ":
            if len(current_token) > 0:
                tokens += [current_token]
                current_token = ""
        elif char == ".":
            if len(current_token) > 0:
                tokens += [current_token]
            tokens += ["."]
            current_token = ""
        else:
            current_token += char
    if len(current_token) > 0:
        tokens += [current_token]
    tokens = split_hyphen_nouns(tokens)
    return [token for token in tokens if keep_token(token)]


def tokens_to_sentences(tokens: List[str]) -> List[List[str]]:
    """
    Finds sentence stops in token list.
    :param tokens: List of tokens
    :return: Creates a list of sentences. Each sentence is a list of tokens
    """
    sentences = []
    current_sentence = []
    for token in tokens:
        if token == ".":
            assert len(current_sentence) > 0
            sentences += [current_sentence]
            current_sentence = []
        else:
            current_sentence += [token]
    if len(current_sentence) > 0:
        sentences += [current_sentence]
    return sentences


def n_gram_count(segment: str, n: int) -> Dict[str, int]:
    n_gram_counts = dict()
    sentences = tokens_to_sentences(tokenize(segment))
    for sentence in sentences:
        n_gram_idx = 0
        while n_gram_idx + n < len(sentence):
            n_gram = tuple([word.lower() for word in sentence[n_gram_idx:n_gram_idx+n]])
            assert len(n_gram) == n
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            n_gram_idx += 1
    return n_gram_counts


def write_vocabulary_by_frequency(vocabulary: Dict[str, int], handle:TextIO,
                                  sorting: str = Literal["frequency", "lexicographic"]):
    if sorting == "frequency":
        vocab_sorted = sorted(vocabulary.items(), key=lambda x: -x[1])
    elif sorting == "lexicographic":
        vocab_sorted = sorted(vocabulary.items(), key=lambda x: x[0])
    for token, frequency in vocab_sorted:
        handle.write(f"{frequency}:\t{token}\n")
