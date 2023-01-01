from typing import Dict, List, Union
import pandas as pd

import tokenizer


def strip_ending_characters(sentence: str) -> str:
    while sentence.endswith(" ") or sentence.endswith("."):
        sentence = sentence[:-1]
    return sentence


def prepare_sentence_list(sentences: List[str], raise_errors: bool = True) -> str:
    sentences = [tokenizer.prepare_segment_for_tokenization(sentence, raise_errors=raise_errors)
                 for sentence in sentences]
    return ". ".join([strip_ending_characters(sentence) for sentence in sentences])


def row_to_dict(row: pd.Series, min_score: int, raise_errors: bool = True) -> dict:
    d = {"score": row["RF-Score"]-min_score, "patient": row["Patient"], "session": row["Session"],
         "segment": prepare_sentence_list(row["Segment_preproc"], raise_errors=raise_errors)}
    if not d["score"] == int(d["score"]):
        raise ValueError(f"Score out of range: {d['score']}")
    else:
        d["score"] = int(d["score"])
    return d


def create_features(df: pd.DataFrame, min_score: int = 0, raise_errors: bool = True) -> List[Dict[str, Union[str, int]]]:
    """
    :param df:
    :param min_score: Lowest score in the data. Scores will get translated so that they start at zero
    :return:
    """
    features = [row_to_dict(row, min_score=min_score, raise_errors=raise_errors) for idx, row in df.iterrows()]
    return features
