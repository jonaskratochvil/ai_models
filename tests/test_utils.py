import pytest, os, sys

# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from models.utils import random_chunks


def test_random_chunks():
    lower, upper = 3, 7
    lst = [i for i in range(23)]
    for i in range(10):
        for chunk in random_chunks(lst, lower, upper):
            assert lower <= len(chunk) <= upper
