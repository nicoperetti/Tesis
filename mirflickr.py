#coding: utf-8
import argparse

import os
from os.path import join, splitext, exists, abspath, dirname

import numpy as np

import json

import spacy

import settings

from utils import get_file_list

nlp = None


def mirflickr_annotations(min_tag_count=50):
    global nlp

    # tags that appear at least 50 times among all images
    fname = join(settings.MIRFLICKR_PATH, "mirflickr25k", "doc", "common_tags.txt")
    with open(fname) as fh:
        frequent_tags = [lin.split()[0] for lin in fh.readlines() if int(lin.split()[1]) >= min_tag_count]
    frequent_tags = sorted(frequent_tags)

    # read images
    imlist, impath = get_file_list(settings.MIRFLICKR_PATH, (".jpg",))
    potential_tags = {"tag2im": {}, "im2tag": {im: [] for im in imlist}}
    relevant_tags = {"tag2im": {}, "im2tag": {im: [] for im in imlist}}

    # read annotations
    flist, fpath = get_file_list(join(settings.MIRFLICKR_PATH, "annotations"), (".txt",))
    flist.remove("README.txt")

    id2im = lambda id_: "im{}.jpg".format(id_)
    im2id = lambda im_: int(im_[2:-4])

    # 24 potential tags
    for f in [f_ for f_ in flist if not f_.endswith("_r1.txt")]:
        tag = splitext(f)[0]
        with open(join(fpath, f)) as fh:
            potential_tags["tag2im"][tag] = sorted([id2im(id_.strip()) for id_ in fh.readlines()])

    for tag, imlist in potential_tags["tag2im"].items():
        for im in imlist:
            potential_tags["im2tag"][im].append(tag)

    # 14 relevant tags
    for f in [f_ for f_ in flist if f_.endswith("_r1.txt")]:
        tag = splitext(f)[0].replace("_r1", "")
        with open(join(fpath, f)) as fh:
            relevant_tags["tag2im"][tag] = sorted(["im{}.jpg".format(id_.strip()) for id_ in fh.readlines()])

    for tag, imlist in relevant_tags["tag2im"].items():
        for im in imlist:
            relevant_tags["im2tag"][im].append(tag)

    potential_tags_ = list(potential_tags["tag2im"].keys())
    potential_images_ = list(set(sum(potential_tags["tag2im"].values(), [])))
    relevant_tags_ = list(relevant_tags["tag2im"].keys())
    relevant_images_ = list(set(sum(relevant_tags["tag2im"].values(), [])))

    print(" >> tags w/ more than {} counts: {}".format(min_tag_count, len(frequent_tags)))
    print(" >> potential tags: {} ({} images)".format(len(potential_tags_), len(potential_images_)))
    print(" >> relevant tags: {} ({} images)".format(len(relevant_tags_), len(relevant_images_)))

    mirflickr = {
        "tags": sorted(list(set(frequent_tags + potential_tags_ + relevant_tags_))),
        "frequent": frequent_tags,
        "potential": potential_tags,
        "relevant": relevant_tags,
    }

    return mirflickr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get mirflickr25k tags",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--html", help="generate HTML visualization of first 100 entries", action="store_true")
    parser.add_argument("--output", help="output file (JSON)", type=str, default="./mirflickr.tags")
    args = parser.parse_args()

    # --------------------------------------------------------------------------

    mirflickr_file = abspath(args.output)
    if not exists(mirflickr_file):
        mirflickr = mirflickr_annotations()
        if not exists(dirname(mirflickr_file)):
            os.makedirs(dirname(mirflickr_file))
        json.dump(mirflickr, open(mirflickr_file, "w"))
        print("{} saved".format(mirflickr_file))
    else:
        mirflickr = json.load(open(mirflickr_file, "r"))
        print("{} already exist".format(mirflickr_file))
