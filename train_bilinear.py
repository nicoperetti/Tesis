# coding: utf-8
import sys

import os
from os.path import splitext, join, exists

import argparse

import numpy as np

import time

import json

from utils import get_file_list, load, normalize_rows, progressbar, timestamp

import torch

import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
GPU_AVAILABLE = torch.cuda.is_available()

from tensorboardX import SummaryWriter

LOGS_DIR = './runs'

CHECKPOINTS_DIR = './checkpoints'

from dataset import PairedTensorDataset

from embedding import Bilinear

from loss import *

from rank_metrics import ndcg_at_k



def compute_metrics(model, criterion, loader, k=5):
    global GPU_AVAILABLE

    loss = 0.
    p_at_1 = 0
    p_at_k = 0
    ndcg = 0

    for X, Y in loader:
        X = Variable(X)
        Y = [Variable(y) for y in Y]
        if GPU_AVAILABLE:
            X = X.cuda()
            Y = [y.cuda() for y in Y]

        outputs = model(X, Y)

        loss += criterion(outputs).data.item()

        if GPU_AVAILABLE:
            outputs = [out.cpu() for out in outputs]

        outputs = [out.data.numpy().squeeze() for out in outputs]

        idxs = [np.argsort(out)[::-1] for out in outputs]
        p_at_1 += sum([np.mean(idx[:1] < 1) for idx in idxs])
        p_at_k += sum([np.mean(idx[:k] < k) for idx in idxs])
        ndcg += sum([ndcg_at_k(out.tolist(), k=k, method=0) for out in outputs])

    N = len(loader.dataset)
    return loss/N, p_at_1/N, p_at_k/N, ndcg/N


def train(X_train, Y_train, X_val, Y_val, param):
    global GPU_AVAILABLE, LOGS_DIR, CHECKPOINTS_DIR

    batch_size = param['batch_size']
    learning_rate = param['learning_rate']
    epochs = param['epochs']
    loss_type = param['loss']
    top_k = param['top_k']

    collate_fn = PairedTensorDataset.collate_fn

    # datasets and data loaders
    X_train = torch.from_numpy(X_train)
    Y_train = [torch.from_numpy(y) for y in Y_train]
    train_data = PairedTensorDataset(X_train, Y_train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    X_val = torch.from_numpy(X_val)
    Y_val = [torch.from_numpy(y) for y in Y_val]
    val_data = PairedTensorDataset(X_val, Y_val)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print('{} training / {} validation samples'.format(len(X_train), len(X_val)))

    # bilinear compatibility model
    dim_X, dim_Y = X_train.shape[1], Y_train[0].shape[1]
    model = Bilinear(in1_features=dim_X, in2_features=dim_Y, bias=True)
    if GPU_AVAILABLE:
        model.cuda()

    # loss
    if loss_type == 'sje' or loss_type == "sje1":
        criterion = SJELoss1()
    elif loss_type == 'sje1sq':
        criterion = SJELoss1Square()
    elif loss_type == 'sje2':
        criterion = SJELoss2()
    elif loss_type == 'ranking':
        criterion = SRankingLoss(top_k)
    else:
        raise ValueError('\'{}\' is not a valid loss'.format(loss_type))

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # # Decay LR by a factor of 0.1 every N epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(LOGS_DIR)

    iter_ = 0

    # Training the Model
    num_epochs = epochs
    for epoch in range(1, num_epochs+1):

        # scheduler.step()

        epoch_loss = 0.

        for X, Y in train_loader:
            X = Variable(X)  #[Variable(x.unsqueeze_(0)) for x in X]
            Y = [Variable(y) for y in Y]

            if GPU_AVAILABLE:
                X = X.cuda()
                Y = [y.cuda() for y in Y]

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(X, Y)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.data.item(), iter_)
            iter_ += 1

            epoch_loss += loss.data.item()

        epoch_loss /= len(X_train)
        print('epoch={}, loss={:.3e}'.format(epoch, epoch_loss))

        # log
        val_loss, p_at_1, p_at_k, ndcg = compute_metrics(model, criterion, val_loader, k=5)
        writer.add_scalar('metrics/p\@1', p_at_1, epoch)
        writer.add_scalar('metrics/p\@5', p_at_k, epoch)
        writer.add_scalar('metrics/nDCG\@5', ndcg, epoch)
        writer.add_scalar('loss/train', epoch_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)

        # checkpoint
        if epoch % 5 == 0:
            model_file = join(CHECKPOINTS_DIR, 'model_{:04d}.pth'.format(epoch))
            torch.save(model, model_file)
            print('checkpoint saved to {}'.format(model_file))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrylevel categories',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda s: s.lower() in ("yes", "true", "t", "1", "on"))
    parser.add_argument("tags", help="json with precomputed tags", type=str)
    parser.add_argument("vectors", help="json with precomputed word embeddings", type=str)
    parser.add_argument("features_path", help="base", type=str)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--loss", help="loss (\"sje\", \"ranking\")", type=str, default="sje")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=30)
    parser.add_argument("--top_k", help="consider only the top-k entries in the annotations list (ranking loss)", type=int, default=5)
    parser.add_argument("--suffix", help="output path suffix", type=str, default="")
    parser.add_argument("--seed", help="random seed", type=int, default=1234)
    parser.add_argument("--debug", help="if set, trains the model using 1000 samples. Usefull for HPO and debugging", action="store_true")
    args = parser.parse_args()

    random_state = np.random.RandomState(args.seed)

    if ".glove" in args.vectors:
        vec_type = "glove"
    elif ".word2vec" in args.vectors:
        vec_type = "word2vec"
    elif ".fasttext" in args.vectors:
        vec_type = "fasttext"
    else:
        raise IOError("not a valid vectors file")

    SUFFIX = "{}_learning_rate_{}_batch_size_{}_epochs_{}_{}".format(vec_type, args.learning_rate, args.batch_size, args.epochs, args.loss)
    if args.loss == "ranking":
        SUFFIX += "_top_k_{}".format(args.top_k)

    if args.suffix != "":
        SUFFIX += "_{}".format(args.suffix)

    OUTPUT_DIR = timestamp(SUFFIX)

    LOGS_DIR = join(LOGS_DIR, OUTPUT_DIR)
    if not exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    CHECKPOINTS_DIR = join(CHECKPOINTS_DIR, OUTPUT_DIR)
    if not exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)

    # load COCO data
    print("loading annotations ... ", end='')
    sys.stdout.flush()
    anno = json.load(open(args.tags, 'r'))
    print("done")

    # load word embeddings
    print("loading word embeddings ... ", end='')
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

    X, Y = {}, {}

    print("loading features ... ", end=''); sys.stdout.flush()
    for set_ in [k for k in anno.keys() if k != "tags"]:
        imid_list = sorted(list(anno[set_].keys()))  # only for reproducibility
        X[set_] = None
        Y[set_] = [None for _ in range(len(imid_list))]

        for i, imid in enumerate(imid_list):
            # set image features
            fname = splitext(anno[set_][imid]["file_name"])[0] + ".dat"
            x = load(join(args.features_path, set_, fname))
            if i == 0:
                n_samples = len(imid_list)
                n_dim = len(x)
                X[set_] = np.empty((n_samples, n_dim), dtype=np.float32)
            X[set_][i] = normalize_rows(x.reshape(1, -1)).squeeze()

            # set word embeddings (OOV tags are set to the zero vector)
            tags = anno[set_][imid]["tags"]
            y = [[0]*vec_dim if vecs[w] is None else vecs[w] for w in tags]
            Y[set_][i] = normalize_rows(np.array(y, dtype=np.float32))

    print("done")

    # shuffle debug
    if args.debug:
        idxs = random_state.permutation(len(X["train2014"]))[:1000]
        X["train2014"] = X["train2014"][idxs]
        Y["train2014"] = [Y["train2014"][i] for i in idxs]

    # run
    param = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "top_k": args.top_k
    }
    train(X["train2014"], Y["train2014"], X["val2014"], Y["val2014"], param)
