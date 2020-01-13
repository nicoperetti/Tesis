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

from utils import get_file_list, load, progressbar, normalize_rows

from rank_metrics import average_precision

nlp = None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run experiments on MIRFlickr25k dataset',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("tags", help="json with precomputed tags", type=str)
    parser.add_argument("vectors", help="json with precomputed word embeddings", type=str)
    parser.add_argument('features_path', help='features path', type=str)
    parser.add_argument('model', help='pretrained model', type=str)
    parser.add_argument('--tag_set', help='test set (\'relevant\', \'potential\', \'all\')', type=str, default='relevant')
    #parser.add_argument('--output', help='output file (JSON)', type=str, default='./data/mirflickr.json')
    args = parser.parse_args()

    # --------------------------------------------------------------------------

    # load annotations
    print("loading annotations ... ", end='')
    sys.stdout.flush()
    annot = json.load(open(args.tags, 'r'))
    print("done")

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

    print("loading model ... ", end='')
    sys.stdout.flush()
    from embedding import *
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    print("done")

    # --------------------------------------------------------------------------

    if args.tag_set in ('relevant', 'potential'):
        tag2im = annot[args.tag_set]['tag2im']
        im2tag = annot[args.tag_set]['im2tag']

    elif args.tag_set == 'all':
        tag2im = annot['potential']['tag2im']
        for tag, imlist in annot['relevant']['tag2im'].items():
            tag2im[tag + '*'] = imlist

        im2tag = annot['potential']['im2tag']
        for im, taglist in annot['relevant']['im2tag'].items():
            if im not in im2tag:
                im2tag[im] = []
            im2tag[im] += [tag + '*' for tag in taglist]

        relevant_tags = list(annot['relevant']['tag2im'].keys())
        annot['tags'] += [tag + '*' for tag in relevant_tags]
        # for tag in relevant_tags:
        #     i = annot['tags'].index(tag)
        #     annot['vectors'].append(annot['vectors'][i])
    else:
        raise ValueError('not a vlid test set')

    assert(len(im2tag.keys()) == 25000)

    # --------------------------------------------------------------------------

    images = ["im{}.jpg".format(i) for i in range(1, 25001)]

    images_train = [images[i] for i in range(0, len(images), 2)]

    images_test = [images[i] for i in range(1, len(images), 2)]

    tags = sorted(list(tag2im.keys()))

    # --------------------------------------------------------------------------

    print("processing input (training) embeddings ...")
    ebd_x = []
    ebd_x_proj = []
    for im in progressbar(images):
        feat_file = join(args.features_path, im.replace('.jpg', '.dat'))
        x = load(feat_file)
        ebd_x.append(x)
        x_proj = model.project_x(torch.from_numpy(x)).squeeze_(0).data.numpy()
        ebd_x_proj.append(x_proj)
    ebd_x = normalize_rows(np.array(ebd_x), 2)
    ebd_x_proj = normalize_rows(np.array(ebd_x_proj), 2)
    print("{} image features".format(len(ebd_x)))

    # --------------------------------------------------------------------------

    print("processing output embeddings ...")
    ebd_y = []
    ebd_y_proj = []
    for tag in progressbar(tags):
        tag = tag.replace('*', '')
        if vecs[tag] is None:
            y = np.zeros(vec_dim, dtype=np.float32)
        else:
            y = np.array(vecs[tag], dtype=np.float32)
        ebd_y.append(y)
        y_proj = model.project_y(torch.from_numpy(y))
        ebd_y_proj.append(y_proj.squeeze_(0).data.numpy())
    ebd_y = normalize_rows(np.array(ebd_y), 2)
    ebd_y_proj = normalize_rows(np.array(ebd_y_proj), 2)
    print("{} tag features".format(len(ebd_y)))

    # --------------------------------------------------------------------------

    avg_feat_train = []
    for tag in tags:
        relevant = [im for im in tag2im[tag] if im in images_train]
        idxs = [images.index(im) for im in relevant]
        avg_feat_train.append(np.mean(np.atleast_2d(ebd_x[idxs]), axis=0).squeeze())
    avg_feat_train = normalize_rows(np.array(avg_feat_train), 2)

    scr = np.zeros((len(images_test), len(tags)))
    for i, im in enumerate(images_test):
        j = images.index(im)
        scr1 = ebd_y_proj.dot(ebd_x[j].reshape(-1, 1)).squeeze()
        scr2 = avg_feat_train.dot(ebd_x[j].reshape(-1, 1)).squeeze()
        #scr[i, :] = 0.5 * (np.maximum(scr1, 0) ** 0.5 + np.maximum(scr2, 0) ** 0.5)
        #scr[i, :] = 0.5 * (np.maximum(scr1, 0) + np.maximum(scr2, 0))
        scr[i, :] = 0.5 * (scr1 + scr2)

    # --------------------------------------------------------------------------

    print("computing tag-centric scores ..")
    ap = []
    for i, tag in enumerate(tags):
        # rank images
        idxs = np.argsort(scr[:, i])[::-1]

        # compute AP(tag)
        relevant = [im for im in tag2im[tag] if im in images_test]
        r = [int(images_test[j] in relevant) for j in idxs]
        ap.append(average_precision(r))

        print("  {} {:.2f}".format(tag, 100*ap[-1]))

    print("done")

    # --------------------------------------------------------------------------

    print("computing image-centric scores ..")
    iap = []
    for i, im in enumerate(images_test):
        # rank tags
        idxs = np.argsort(scr[i, :])[::-1]

        # compute AP(image)
        relevant = list(im2tag[im])
        r = [int(tags[j] in relevant) for j in idxs]
        iap.append(average_precision(r))
    print("done")

    print("\nMAP: {:.2f} ({} tags)".format(100 * np.mean(ap), len(ap)))
    print("MiAP: {:.2f} ({} images)".format(100 * np.mean(iap), len(iap)))
