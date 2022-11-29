from typing import Tuple, Set, Dict
import pandas as pd
import numpy as np

import tokenizer
import utils


def create_features(df: pd.DataFrame, n_gram_length: int,
                    n_gram_selection: Set[Tuple[str]] = None) -> Tuple[pd.DataFrame, Set[Tuple[str]]]:
    """

    :param df:
    :param n_gram_selection: If present, only the specified n-grams will be considered. This is used for prediction
     on new data
    :return: The features as a dataframe,
    """
    if n_gram_selection is None:
        n_gram_selection = count_all_n_grams(df, n_gram_length)

    all_samples = []
    # each row in the dataframe is a sample for training/testing
    # determine the frequency for all n-grams in the selection
    for idx, segment in df.iterrows():
        n_gram_counts = count_n_grams(n_gram_length, n_gram_selection, segment)

        segment_df = df.loc[[idx]]
        feature_df = n_gram_counts_to_df(df, n_gram_counts, segment_df)
        feature_df = normalize_count_vector(feature_df)
        # Prepend the original sample info (subject, block etc.). This would be removed if we were to use
        # subject, session etc. as indices and remove all other columns
        feature_df = pd.concat([segment_df, feature_df], axis=1, ignore_index=False)
        all_samples += [feature_df]
    return pd.concat(all_samples, axis=0, ignore_index=True), n_gram_selection


def normalize_count_vector(feature_df: pd.DataFrame) -> pd.DataFrame:
    # normalize word counts to compensate for different lengths of segments
    df_sum = feature_df.sum(axis=1)
    if np.all(df_sum > 0):
        feature_df = feature_df.div(df_sum, axis=0)
    assert np.all(feature_df >= 0)
    # NaNs should not occur because we added zero-count entries earlier
    assert np.all(np.isfinite(feature_df))
    return feature_df


def count_all_n_grams(df: pd.DataFrame, n_gram_length: int) -> Set[Tuple[str]]:
    # find all n-grams occurring in the dataframe
    n_gram_selection = set()
    for idx, segment in df.iterrows():
        for line in segment["Segment_preproc"]:
            line_counts = tokenizer.n_gram_count(line, n=n_gram_length)
            n_gram_selection.update(line_counts.keys())
    return n_gram_selection


def n_gram_counts_to_df(df: pd.DataFrame, n_gram_counts: int, segment_df: pd.DataFrame) -> pd.DataFrame:
    # convert all dictionary entries into dataframes
    feature_df = []
    for n_gram, count in n_gram_counts.items():
        # N-grams are represented as tuples up until now. Tuples would be interpreted as
        # multidimensional index by pandas. This is the reason for converting to str,
        # with words being separated by whitespaces again
        assert not any([" " in token for token in n_gram])
        df_key = " ".join(n_gram)
        # to avoid this potential problem, we could use subject, session etc. as index and remove all
        # other columns
        if df_key in df.columns:
            raise NameError(f"Name conflict between n-gram and dataframe column name: {df_key}")
        feature_df += [pd.DataFrame(index=segment_df.index, data={df_key: count})]
    # concatenate the converted frequency count dataframes
    feature_df = pd.concat(feature_df, axis=1, ignore_index=False)
    return feature_df


def count_n_grams(n_gram_length: int, n_gram_selection: Set[Tuple[str]], segment: pd.Series) -> Dict[Tuple[str], int]:
    n_gram_counts = dict()
    # first, count the n-grams from the selection which do exist in the current row
    for line in segment["Segment_preproc"]:
        line_counts = {n_gram: count
                       for n_gram, count in tokenizer.n_gram_count(line, n=n_gram_length).items()
                       if n_gram in n_gram_selection}
        utils.add_dicts(n_gram_counts, line_counts)
    # then add entries with count 0 for all n-grams from the selection which do not occur in the current row
    zero_counts = {n_gram: 0 for n_gram in n_gram_selection if n_gram not in n_gram_counts}
    n_gram_counts.update(zero_counts)
    assert set(n_gram_counts.keys()) == n_gram_selection
    return n_gram_counts
