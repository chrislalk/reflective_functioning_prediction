import data_reader
import tokenizer

import sys


class Pipeline(object):
    def __init__(self, data_file):
        self.data_file = data_file
        self.full_data = None
        self.vocabulary = None

    def run(self) -> None:
        self.full_data = data_reader.read_full_dataset(self.data_file)
        self.vocabulary = tokenizer.build_vocabulary(self.full_data["Segment_preproc"])


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx")
    pipeline.run()
    tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
