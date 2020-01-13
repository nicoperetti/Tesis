# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
from os.path import join, splitext, abspath, exists, commonpath

import numpy as np

from nltk.corpus import wordnet as wn

from tqdm import tqdm

from itertools import combinations, product

from datetime import datetime

import scipy.io as sio


def save(filename, mat):
    """ saves data in binary format
    Args:
       filename (str): file name
       mat (ndarray): numpy array
    """
    if not isinstance(mat, np.ndarray):
        raise ValueError('for now, we can only save numpy arrays')
    return sio.savemat(filename, {'data': mat}, appendmat=False)


def load(filename):
    """ load data from file
    Args:
       filename (str): file name

    Returns:
      loaded array
    """
    return sio.loadmat(filename, appendmat=False, squeeze_me=True)['data']


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def timestamp(suffix=None):
    fmt = '%Y%m%d_%H%M%S'
    #fmt = '%Y%m%d'
    if suffix is not None:
        fmt += "_{}".format(suffix)
    return datetime.now().strftime(fmt)


# wnid -> WordNet synset
def wnid2synset(wnids):
    _wnid2synset = lambda id: wn.of2ss(id[1:] + id[0])
    if isinstance(wnids, (tuple, list)):
        return [_wnid2synset(id) for id in wnids]
    return _wnid2synset(wnids)


# WordNet synset -> wnid
def synset2wnid(s):
    return '{}{:08d}'.format(s.pos(), s.offset())


def normalize_rows(mat, ord=2):
    ''' return a row normalized matrix
    '''
    assert mat.ndim == 2
    norms = zeros_to_eps(np.linalg.norm(mat, ord=ord, axis=1))
    return mat / norms.reshape(-1, 1)


def zeros_to_eps(mat):
    ''' replace zeros in a matrix by a tiny constant
    '''
    mat[np.isclose(mat, 0.)] = np.finfo(mat.dtype).eps
    return mat


def get_file_list(path, valid_extensions=None):
    """
    Args:
       path (str): base path
       valid_extensions (list): list of valid extensions
    Returns:
       flist, base_path: file list and base path, so that
          join(base_path, flist[i]) gives the full path of the i-th file
    """
    if valid_extensions is None:
        is_valid = lambda f: True
    else:
        is_valid = lambda f: bool(splitext(f)[1].lower() in valid_extensions)

    path = abspath(path)
    if not exists(path):
        raise OSError('{} doesn\'t exist'.format(path))

    # get file list
    flist = []
    for root, _, files in os.walk(path, followlinks=True):
        for fname in [f for f in files if is_valid(f)]:
            flist.append(join(root, fname))

    if len(flist) == 0:
        return [], path

    # relative path
    fpath = commonpath(flist) + '/'
    flist = [f.replace(fpath, '') for f in flist]

    return flist, fpath


def sample_unrelated_inplace(data, n_samples=100):
    """
    Args:
       data (dict): precomputed tags
       n_samples (int): number of unrelated tags per sample
    Returns:
       None
    """

    tags = set(data["tags"])
    for set_ in data.keys():
        if set_ == "tags":
            continue

        print("sampling unrelated tags for the \"{}\" set ...".format(set_))
        for imid, data_ in progressbar(data[set_].items()):
            tags_ = set(data_["tags"])

            unrelated = tags.difference(tags_)
            unrelated = np.random.choice(list(unrelated),
                                         min(n_samples, len(unrelated)),
                                         replace=False).tolist()
            data[set_][imid]["unrelated"] = unrelated
    return


def check_hyper(words):
    """
    Check if one word ins hyperomin of the other.

    Args:
        words (list): a list of words to be checked

    Return:
        bool
    """
    check = False
    if len(words) > 1:
        check = True
        for word1, word2 in combinations(words, 2):
            sysn1 = wn.synsets(word1)
            sysn2 = wn.synsets(word2)
            check_hyp = False
            for sys1, sys2 in product(sysn1, sysn2):
                if (sys2 in sys1.common_hypernyms(sys2)) or \
                   (sys1 in sys2.common_hypernyms(sys1)):
                    check_hyp = True
                    break
            if not check_hyp:
                check = False
                break
    return check
