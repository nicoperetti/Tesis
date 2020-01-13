#coding: utf-8

import argparse

import os
from os.path import join, splitext, exists, dirname, abspath, commonpath

from PIL import Image

from feature_extractor import *

from utils import progressbar, get_file_list, save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute image features keeping the dataset path tree',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda s: s.lower() in ("yes", "true", "t", "1", "on"))
    parser.add_argument('dataset', help='dataset base path', type=str)
    parser.add_argument('--features', help='feature type (vgg16, vgg19, resnet50, resnet152, rmac)', type=str, default='vgg16')
    parser.add_argument('--output_path', help='output path', type=str, default='./data')
    args = parser.parse_args()

    # get image list from path
    imlist, impath = get_file_list(args.dataset, ('.jpg', '.jpeg', '.png') )
    print('{} images'.format(len(imlist)))

    imlist = sorted(imlist)

    if args.features == 'vgg16':
        model = VGG16FeatureExtractor()
    elif args.features == 'vgg19':
        model = VGG19FeatureExtractor()
    elif args.features == 'resnet50':
        model = ResNet50FeatureExtractor()
    elif args.features == 'resnet152':
        model = ResNet152FeatureExtractor()
    elif args.features == 'rmac':
        model = RMACFeatureExtractor()
    else:
        raise RuntimeError('not a valid feature type')

    for imfile in progressbar(imlist):
        featfile = join(args.output_path, splitext(imfile)[0]) + '.dat'
        if exists(featfile):
            continue

        img = Image.open(join(impath, imfile)).convert('RGB')
        feat = model.process(img)

        if not exists(dirname(featfile)):
            os.makedirs(dirname(featfile))

        save(featfile, feat)

        # print('{} ({}x{}): {} features'.format(imfile, im.size[0], im.size[1], len(feat)))
