import utils
import data_reader
import tokenizer

from typing import List, Tuple, Iterable, Set
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import numpy as np
import os


class Pipeline(object):
    def __init__(self, data_file, n_gram_length, output_dir):
        self.data_file = data_file
        self.n_gram_length = n_gram_length
        self.output_dir = output_dir
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

    @staticmethod
    def init_classifier():
        """
        Create classifier here and set its parameters
        """
        return sklearn.svm.SVR()

    def run(self) -> None:
        self.full_data = data_reader.read_full_dataset(self.data_file)
        train_test_splits = Pipeline.get_train_test_splits(self.full_data)
        result_df = []
        for train, test in train_test_splits:
            training_data = self.full_data.loc[self.full_data["Patient"].isin(train)]
            testing_data = self.full_data.loc[self.full_data["Patient"].isin(test)]

            # get training features and a list of n-grams which occurred in training data
            training_features, selected_n_grams = Pipeline.create_features(training_data,
                                                                           n_gram_length=self.n_gram_length)
            # use this list to create testing features
            testing_features, _ = Pipeline.create_features(testing_data, n_gram_length=self.n_gram_length,
                                                           n_gram_selection=selected_n_grams)
            # reorder testing features
            assert set(training_features.columns) == set(testing_features.columns)
            testing_features = testing_features[training_features.columns]
            assert all(training_features.columns == testing_features.columns)

            # create and train classifier
            classifier = Pipeline.init_classifier()
            classifier.fit(X=training_features[[" ".join(n_gram) for n_gram in selected_n_grams]],
                           y=training_features["RF-Score"])

            # obtain predictions and score
            predictions = classifier.predict(X=testing_features[[" ".join(n_gram) for n_gram in selected_n_grams]])
            mae = np.mean(np.abs(testing_data["RF-Score"] - predictions))
            result_df += [pd.DataFrame(index=[0], data={"test": [test], "train": [train],
                                                        "actual": [testing_data["RF-Score"].values],
                                                        "predicted": [predictions],
                                                        "MAE": [mae]})]
        result_df = pd.concat(result_df, ignore_index=True, axis=0)
        print(result_df)
        output_file = os.path.join(self.output_dir, "cv_results.txt")
        result_df.to_csv(output_file, sep="\t")


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx", n_gram_length=3, output_dir=r"..\output")
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
