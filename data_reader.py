import re

import pandas as pd
from typing import List


def keep_line(line: str) -> bool:
    """
    Returns a boolean indicating whether a line should be kept in the text segment
    Removes brackets and therapist utterances
    """
    ls = line.strip()
    if len(ls) == 0:
        return False
    if ls.startswith("{") and ls.endswith("}"):
        return False
    elif ls.startswith("[") and ls.endswith("]"):
        return False
    if not (ls.startswith("T:") or ls.startswith("P:")):
        raise ValueError(f"Unknown line type (who is speaking?): {ls}")
    if ls.startswith("T:"):
        return False
    return True


def strip_segment(segment: str) -> List[str]:
    """
    Strip empty lines and metadata
    :return: line-wise split of segment data, removed non-analyzable lines
    """
    lines = [line.strip() for line in segment.replace("\r", "\n").split("\n") if keep_line(line)]
    return lines


def read_full_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Segment_preproc"] = [strip_segment(segment) for segment in df["Segment"]]
    df[["Patient", "Session", "Dokument_Residual"]] = df["Dokumentname"].str.split("_", expand=True)
    if not all(valid := df["Patient"].str.match(pat="^[a-zA-Z0-9öäüßÖÄÜ]+$", case=False)):
        invalid_values = df.loc[~valid, "Dokumentname"]
        raise NameError('Unexpected values for "Dokumentname":\n' + "\n".join(invalid_values))
    if not all(valid := df["Session"].str.match(pat="^[0-9]+$", case=False)):
        invalid_values = df.loc[~valid, "Dokumentname"]
        raise NameError('Unexpected values for "Dokumentname":\n' + "\n".join(invalid_values))
    if not all(valid := df["Dokument_Residual"].str.match(pat="^Blöcke$", case=False)):
        invalid_values = df.loc[~valid, "Dokumentname"]
        raise NameError('Unexpected values for "Dokumentname":\n' + "\n".join(invalid_values))
    return df


if __name__ == "__main__":
    full_data = read_full_dataset(r"..\data\Beispiel-Segmente CRF.xlsx")
    print(full_data)
    print(full_data.loc[2, "Segment"])
    print(full_data.loc[2, "Segment_preproc"])
