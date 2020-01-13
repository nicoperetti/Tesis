 #coding: utf-8
import argparse
import sys
import os
from os.path import join, splitext, exists, abspath, dirname

import numpy as np

import json

import spacy

import settings

import torch
import torch.utils.data as data
from torch.autograd import Variable
GPU_AVAILABLE = torch.cuda.is_available()

from dataset import PairedTensorDataset

from utils import get_file_list, load, progressbar, normalize_rows

from rank_metrics import ndcg_at_k

nlp = None


def compute_metrics(model, loader, k=5, mode='bilinear'):
    global GPU_AVAILABLE

    p_at_1 = 0
    p_at_k = 0
    ndcg = 0

    for X, Y in loader:
        X = Variable(X)
        Y = [Variable(y) for y in Y]
        if GPU_AVAILABLE:
            X = X.cuda()
            Y = [y.cuda() for y in Y]

        if mode == "bilinear":
            outputs = model(X, Y)
            if GPU_AVAILABLE:
                outputs = [out.cpu() for out in outputs]
            outputs = [out.data.numpy().squeeze() for out in outputs]

        elif mode == "project_x":
            X_proj = model.project_x(X).data.numpy()
            X_proj = normalize_rows(X_proj)
            Y = [y.data.numpy() for y in Y]
            outputs = [x.reshape(1, -1).dot(np.atleast_2d(y).T).squeeze() for x, y in zip(X_proj, Y)]

        elif mode == "project_y":
            Y_proj = [model.project_y(y).data.numpy() for y in Y]
            Y_proj = [normalize_rows(y) for y in Y_proj]
            X = X.data.numpy()
            outputs = [x.reshape(1, -1).dot(np.atleast_2d(y).T).squeeze() for x, y in zip(X, Y_proj)]

        elif mode == "random":
            outputs = [np.random.random(len(y)) for y in Y]

        else:
            raise ValueError("not a valid mode")

        idxs = [np.argsort(out)[::-1] for out in outputs]
        p_at_1 += sum([np.mean(idx[:1] < 1) for idx in idxs])
        p_at_k += sum([np.mean(idx[:k] < k) for idx in idxs])
        ndcg += sum([ndcg_at_k(out.tolist(), k=k, method=0) for out in outputs])

    N = len(loader.dataset)
    return p_at_1/N, p_at_k/N, ndcg/N


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run experiments on the pascal-sentences dataset',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("tags", help="json with precomputed tags", type=str)
    parser.add_argument("vectors", help="json with precomputed word embeddings", type=str)
    parser.add_argument('features_path', help='path to visual features', type=str)
    parser.add_argument('model', help='pretrained model', type=str)
    parser.add_argument('--n_unrelated', help='number of unrelated samples', type=int, default=0)
    #parser.add_argument('--output', help='output file (JSON)', type=str, default='./data/mirflickr.json')
    args = parser.parse_args()

    # --------------------------------------------------------------------------

    # load annotations
    print("loading annotations ... ", end=' ')
    sys.stdout.flush()
    anno = json.load(open(args.tags, 'r'))
    print("done")

    # --------------------------------------------------------------------------

    # load word embeddings
    print("loading word embeddings ... ", end=' ')
    sys.stdout.flush()
    vecs = json.load(open(args.vectors, 'r'))
    print("done")

    vec_dim = -1
    for w, vec in vecs.items():
        if vec is not None:
            vec_dim = len(vec)
            break
    if vec_dim is None:
        raise RuntimeError("couln'\t set embeddings dimensionality")

    # --------------------------------------------------------------------------

    print("loading model ... ", end=' ')
    sys.stdout.flush()
    from embedding import *
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    print("done")

    X, Y = {}, {}

    # --------------------------------------------------------------------------

    print("loading features ... ", end=' ')
    sys.stdout.flush()

    im_list = sorted(list(anno['pascal'].keys()))  # only for reproducibility
    X = None
    Y = [None for _ in range(len(im_list))]

    for i, im in enumerate(im_list):
        # set image features
        fname = splitext(im)[0] + ".dat"
        x = load(join(args.features_path, fname))
        if i == 0:
            n_samples = len(im_list)
            n_dim = len(x)
            X = np.empty((n_samples, n_dim), dtype=np.float32)
        X[i, :] = normalize_rows(x.reshape(1, -1)).squeeze()

        # set word embeddings (OOV tags are set to the zero vector)
        tags = anno['pascal'][im]["tags"] + anno['pascal'][im]["unrelated"][:args.n_unrelated]
        y = [[0]*vec_dim if vecs[w] is None else vecs[w] for w in tags]
        Y[i] = normalize_rows(np.array(y, dtype=np.float32))
    print("done")


    # datasets and data loaders
    X = torch.from_numpy(X)
    Y = [torch.from_numpy(y) for y in Y]
    dataset = PairedTensorDataset(X, Y)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PairedTensorDataset.collate_fn)

    for k in [1, 3, 5, 7, 9]:
        _, p, ndcg = compute_metrics(model, loader, k=k, mode="bilinear")
        _, p_rnd, _ = compute_metrics(model, loader, k=k, mode="random")
        print("p@{}={:.4f} (RND={:.4f}), nDCG@{}={:.4f}".format(k, p, p_rnd, k, ndcg))
