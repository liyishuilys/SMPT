

"""
| Splitters
"""

import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


__all__ = [
    'RandomSplitter',
    'IndexSplitter',
    'ScaffoldSplitter',
]


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test 
    subsets.
    """
    def __init__(self):
        super(Splitter, self).__init__()


class RandomSplitter(Splitter):
    """
    Random splitter.
    """
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    """
    Split daatasets that has already been orderd. The first `frac_train` proportion
    is used for train set, the next `frac_valid` for valid set and the final `frac_test` 
    for test set.
    """
    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the 
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset[ind]['smiles'], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
        