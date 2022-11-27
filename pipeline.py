import data_reader
import tokenizer

from typing import List, Tuple, Iterable
import sklearn.model_selection
import pandas as pd
import numpy as np
import sys


class Pipeline(object):
    def __init__(self, data_file):
        self.data_file = data_file
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

    def run(self) -> None:
        self.full_data = data_reader.read_full_dataset(self.data_file)
        train_test_splits = Pipeline.get_train_test_splits(self.full_data)
        for train, test in train_test_splits:
            training_data = self.full_data.loc[self.full_data["Patient"].isin(train)]
            print(training_data)
            testing_data = self.full_data.loc[self.full_data["Patient"].isin(test)]
            print(testing_data)

        #self.vocabulary = tokenizer.build_vocabulary(self.full_data["Segment_preproc"])


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx")
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
