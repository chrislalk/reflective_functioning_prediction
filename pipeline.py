import data_reader
import features
import tokenizer
import utils

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
    def init_classifier():
        """
        Create classifier here and set its parameters
        """
        return sklearn.svm.SVR()

    def export_all_n_grams(self) -> None:
        all_n_grams = features.list_all_n_grams(self.full_data, n_gram_length=self.n_gram_length)
        all_counts = dict()
        for idx, segment in self.full_data.iterrows():
            segment_counts = features.count_n_grams(n_gram_length=self.n_gram_length, n_gram_selection=all_n_grams,
                                                    segment=segment)
            all_counts = utils.add_dicts(all_counts, segment_counts)
        output_file = os.path.join(self.output_dir, "all_n_grams.txt")
        with open(output_file, "w") as f:
            tokenizer.write_vocabulary_by_frequency(all_counts, f, sorting="frequency")

    def run(self) -> None:
        self.full_data = data_reader.read_full_dataset(self.data_file)
        self.export_all_n_grams()
        train_test_splits = Pipeline.get_train_test_splits(self.full_data)
        result_df = self.predict_validate(train_test_splits)
        print(result_df)
        output_file = os.path.join(self.output_dir, "cv_results.txt")
        result_df.to_csv(output_file, sep="\t", index=False)

    def predict_validate(self, train_test_splits: Iterable[Tuple[List[str], List[str]]]) -> pd.DataFrame:
        result_df = []
        for train, test in train_test_splits:
            training_data = self.full_data.loc[self.full_data["Patient"].isin(train)]
            testing_data = self.full_data.loc[self.full_data["Patient"].isin(test)]

            selected_n_grams, testing_features, training_features = self.create_features(testing_data, training_data)

            result_df += [Pipeline.score_result(selected_n_grams, test, testing_data, testing_features,
                                                train, training_features)]
        result_df = pd.concat(result_df, ignore_index=True, axis=0)
        return result_df

    @staticmethod
    def score_result(selected_n_grams: Set[Tuple[str]], test: List[str], testing_data: pd.DataFrame,
                     testing_features: pd.DataFrame, train: List[str], training_features: pd.DataFrame) -> pd.DataFrame:
        # create and train algorithm
        algorithm = Pipeline.init_classifier()
        algorithm.fit(X=training_features[[" ".join(n_gram) for n_gram in selected_n_grams]],
                      y=training_features["RF-Score"])
        # obtain predictions and score
        predictions = algorithm.predict(X=testing_features[[" ".join(n_gram) for n_gram in selected_n_grams]])
        mae = np.mean(np.abs(testing_data["RF-Score"] - predictions))
        result = pd.DataFrame(index=[0], data={"test": [test], "train": [train],
                                               "actual": [testing_data["RF-Score"].values],
                                               "predicted": [predictions],
                                               "MAE": [mae]})
        return result

    def create_features(self, testing_data: pd.DataFrame, training_data: pd.DataFrame) \
            -> Tuple[Set[Tuple[str]], pd.DataFrame, pd.DataFrame]:
        # get training features and a list of n-grams which occurred in training data
        training_features, selected_n_grams = features.create_features(training_data,
                                                                       n_gram_length=self.n_gram_length)
        # use this list to create testing features
        testing_features, _ = features.create_features(testing_data, n_gram_length=self.n_gram_length,
                                                       n_gram_selection=selected_n_grams)
        # reorder testing features
        assert set(training_features.columns) == set(testing_features.columns)
        testing_features = testing_features[training_features.columns]
        assert all(training_features.columns == testing_features.columns)
        return selected_n_grams, testing_features, training_features


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx", n_gram_length=3, output_dir=r"..\output")
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
