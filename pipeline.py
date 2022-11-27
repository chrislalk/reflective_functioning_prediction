import utils
import data_reader
import tokenizer

from typing import List, Tuple, Iterable, Set
import sklearn.model_selection
import pandas as pd
import numpy as np
import sys


class Pipeline(object):
    def __init__(self, data_file, n_gram_length):
        self.data_file = data_file
        self.n_gram_length = n_gram_length
        self.full_data = None
        self.vocabulary = None

    @staticmethod
    def get_train_test_splits(full_data: pd.DataFrame) -> Iterable[Tuple[List[str], List[str]]]:
        """
        Return validation folds on patient level
        :param full_data: Dataset to be used
        :return:
        """
        patients = full_data["Patient"].unique()
        for train, test in sklearn.model_selection.LeaveOneOut().split(patients):
            yield patients[train], patients[test]

    @staticmethod
    def create_features(df: pd.DataFrame, n_gram_length, n_gram_selection: Set[Tuple[str]] = None) -> Tuple[pd.DataFrame, Set[Tuple[str]]]:
        """

        :param df:
        :param n_gram_selection: If present, only the specified n-grams will be considered. This is used for prediction
         on new data
        :return: The features as a dataframe,
        """
        if n_gram_selection is None:
            # find all n-grams occurring in the dataframe
            n_gram_selection = set()
            for idx, segment in df.iterrows():
                for line in segment["Segment_preproc"]:
                    line_counts = tokenizer.n_gram_count(line, n=n_gram_length)
                    n_gram_selection.update(line_counts.keys())

        all_samples = []
        # each row in the dataframe is a sample for training/testing
        # determine the frequency for all n-grams in the selection
        for idx, segment in df.iterrows():
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

            segment_df = df.loc[[idx]]
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
            # first, concatenate the converted frequency count dataframes
            feature_df = pd.concat(feature_df, axis=1, ignore_index=False)
            # NaNs should not occur because we added zero-count entries earlier
            assert np.all(np.isfinite(feature_df))
            # then, prepend the original sample info (subject, block etc.). This would be removed if we were to use
            # indices (see above)
            feature_df = pd.concat([segment_df, feature_df], axis=1, ignore_index=False)
            all_samples += [feature_df]
        return pd.concat(all_samples, axis=0, ignore_index=True), n_gram_selection

    def run(self) -> None:
        self.full_data = data_reader.read_full_dataset(self.data_file)
        train_test_splits = Pipeline.get_train_test_splits(self.full_data)
        for train, test in train_test_splits:
            training_data = self.full_data.loc[self.full_data["Patient"].isin(train)]
            testing_data = self.full_data.loc[self.full_data["Patient"].isin(test)]

            # get training features and a list of n-grams which occurred in training data
            training_features, selected_n_grams = Pipeline.create_features(training_data,
                                                                           n_gram_length=self.n_gram_length)
            # use this list to create testing features
            testing_features, _ = Pipeline.create_features(testing_data, n_gram_length=self.n_gram_length,
                                                           n_gram_selection=selected_n_grams)
            assert set(training_features.columns) == set(testing_features.columns)


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx", n_gram_length=1)
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
