from typing import Type, List, Dict, Literal, TextIO
import warnings
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
REPLACE_TO_DOTS = [
    # since we do not know how many words are missing when transcribers could not understand something:
    re.compile(r"\(\s*unverständlich\s*\)"),
    re.compile(r"/"),
    re.compile(r"\?"),
    re.compile(r"\!"),
]
_RE_VALID_WORD = re.compile(r"^[äöüßÄÖÜa-zA-Z]*[0-9]*\.?$")
_RE_HYPHENATED_WORD = re.compile(r"^[äöüßÄÖÜa-zA-Z]+[-[äöüßÄÖÜa-zA-Z]+]*$")
_RE_END_MULTIPLE_HYPHENS = re.compile(f"^([^-]+)-+-$")

#IGNORE_WORDS = list(utils.read_file_contents_lowercase("ignore_words.txt"))


def error_handler(msg: str, raise_errors: bool = True, errortype: Type[BaseException] = ValueError):
    if raise_errors:
        raise errortype(msg)
    else:
        warnings.warn(msg)


def keep_token(token: str, raise_errors: bool = True) -> bool:
    """
    Returns a boolean indicating whether a token should be kept
    """
    if len(token) == 0:
        return True
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
    #if token.lower() in IGNORE_WORDS:
    #    return False
    if _RE_HYPHENATED_WORD.match(token) is None:
        # word does not match rule
        # see if it is an anonymized name/place etc.
        if (token.endswith("*") or token.endswith(".") or token[-1].isnumeric()) \
                and keep_token(token[:-1], raise_errors=raise_errors):
            pass
        elif token.isnumeric() or token == ".":
            pass
        else:
            error_handler(f"Not a word: {token}", raise_errors)
    return True


def strip_brackets(segment: str, start_char: str = "(", end_char: str = ")", raise_errors: bool = True) -> str:
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
                error_handler("Found closing bracket without opening bracket", raise_errors)
        elif open_brackets == 0:
            non_bracket_text += char
    if open_brackets > 0:
        error_handler("Opening bracket was never closed", raise_errors)
    return non_bracket_text


def process_word(word: str) -> str:
    """
    Transforms a word, in necessary
    """
    if (match := re.match(string=word, pattern=_RE_END_MULTIPLE_HYPHENS)) is not None:
        non_whitespace = match.group(1)
        assert not non_whitespace.endswith("-")
        return non_whitespace + "."
    return word


def prepare_segment_for_tokenization(segment: str, raise_errors: bool = True) -> str:
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
    # replace dashes with doulbe hyphens
    segment = segment.replace("–", "--")
    segment = segment.replace("—", "--")
    # replace some text with a dot
    for regexp in REPLACE_TO_DOTS:
        segment = regexp.sub(". ", segment)
    # strip brackets
    for start_char, end_char in [["(", ")"],
                                 ["[", "]"],
                                 ["{", "}"]]:
        segment = strip_brackets(segment, start_char=start_char, end_char=end_char, raise_errors=raise_errors)
    # insert whitespace after "P:" if it is not present
    if len(segment) > 2 and segment[1] == ":" and not segment[2] == " ":
        segment = segment[:2] + " " + segment[2:]
    # split by whitespaces, transform
    words = [process_word(word) for word in segment.split(" ")]
    words = [word for word in words if keep_token(word, raise_errors=raise_errors)]
    segment = " ".join(words)
    return segment
