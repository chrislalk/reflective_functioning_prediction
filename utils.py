from typing import Dict, Tuple


def add_dicts(a: Dict[Tuple[str], int], b: Dict[Tuple[str], int]) -> Dict[Tuple[str], int]:
    """
    Adds to dictionary `a` the values of dictionary `b`. Both dicts have tuples of strings as indices and integers as values.
    :param a: Reference to the dictionary which will be changed
    :param b: Dictionary whose values will be added to dictionary `a`.
    :return: Dictionary a
    """
    for key, value in b.items():
        a[key] = a.get(key, 0) + value
    return a
