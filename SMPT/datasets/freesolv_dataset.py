

"""
Processing of freesolv dataset.

The Free Solvation Dataset provides rich information. It contains calculated values and experimental values about hydration free energy of small molecules in water.You can get the calculated values by  molecular dynamics simulations,which are derived from alchemical free energy calculations. However,the experimental values are included in the benchmark collection.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from datasets.inmemory_dataset import InMemoryDataset


def get_default_freesolv_task_names():
    """Get that default freesolv task names and return measured expt"""
    return ['expt']


def load_freesolv_dataset(data_path, task_names=None):
    """Load freesolv dataset,process the input information and the featurizer.
    
    Description:
        
        The data file contains a csv table, in which columns below are used:
            
            smiles: SMILES representation of the molecular structure
            
            Compound ID: Name of the compound
            
            measured log solubility in mols per litre: Log-scale water solubility of the compound, used as label.
   
    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_freesolv_dataset('./freesolv')
            print(len(dataset))

    References:
    
    [1] Mobley, David L., and J. Peter Guthrie. "FreeSolv: a database of experimental and calculated hydration free energies, with input files." Journal of computer-aided molecular design 28.7 (2014): 711-720.
    
    [2] https://github.com/MobleyLab/FreeSolv

    """
    if task_names is None:
        task_names = get_default_freesolv_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_freesolv_stat(data_path, task_names):
    """Return mean and std of labels"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }
