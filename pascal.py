#coding: utf-8
import argparse

from os import makedirs

from os.path import join, exists, splitext, split, dirname, abspath, basename

from urllib.parse import quote

import settings

from utils import progressbar, sample_unrelated_inplace

from tags import NounTags, CompoundTags

from bs4 import BeautifulSoup

import requests

import json

import spacy

import HTML

URL = 'http://vision.cs.uiuc.edu/pascal-sentences/'


def pascal_tags(extractor, dct):
    res = {
        'pascal': {},
        'tags': []
    }

    print('Computing tags ...')
    for im, cpt in progressbar(dct.items()):
        # tags and frequency counts
        tags, scores = extractor.process(cpt)

        res['pascal'][im] = {
            'file_name': im,
            'category_names': [split(im)[0],],
            'captions': cpt,
            'tags': tags,
            'scores': scores,
        }
    print('\ndone')

    for imdata in res['pascal'].values():
        res['tags'] += [w for w in imdata['tags']]
    res['tags'] = list(set(res['tags']))

    return res


def htmlize(data, output_file, n_samples=100):
    """ writes an HTML file for visualization

    Args:
       data (dict): results generated by pascal_tags
       output_file (str): destination file
       n_samples (int): visualize first `n_samples` samples
    """
    if 'pascal' in data:
        data = data['pascal']

    tbl = HTML.Table(header_row=['image', 'category_names', 'captions', 'tags'])
    def html_img(text, url):
        if len(text) > 48:
            text = text[:20] + '...' + text[-24:]
        return '<center>{}</center><img src={} width=320>'.format(text, url)

    for im in sorted(list(data.keys()))[::len(data)//n_samples]:
        html_category_names = '<br>'.join(data[im]['category_names'])
        html_captions = '<br>'.join(data[im]['captions'])

        # tag-score pairs
        tags = data[im]['tags']
        scores = data[im]['scores']
        ts = [(tags[i], scores[i]) for i in range(len(tags))]
        html_tags = '<br>'.join(['{:20s}: {:.3f}'.format(t, s) for t, s in ts])

        fname = join(abspath(settings.PASCAL_PATH), data[im]['file_name'])
        tbl.rows.append([html_img(basename(fname), fname),
                         html_category_names,
                         html_captions,
                         html_tags])

    htmlcode = str(tbl)
    f = open(output_file, 'w')
    f.write(htmlcode)
    f.write('<p>')
    f.close()
    print('{} saved'.format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pascal tags from captions',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--html', help='generate HTML visualization of first 100 entries', action='store_true')
    parser.add_argument('--mode', help='noun / compound', type=str, default='noun')
    parser.add_argument('--unrelated', help='sample unrelated tags', action='store_true')
    parser.add_argument('--syntactic', help='enable syntactic mode', action='store_true')
    parser.add_argument('--alpha', help='alpha', type=float, default=0.99)
    parser.add_argument('--output', help='output file (JSON)', type=str, default='./pascal.tags')
    args = parser.parse_args()

    # get gaptions from HTML
    annotations = join(split(args.output)[0], 'pascal.annotations')
    if not exists(annotations):
        data = requests.get(URL).text
        soup = BeautifulSoup(data, "html.parser")

        annot = {}
        for tag in soup.findAll('img'):
            img = tag.get('src')

            captions = tag.find_parent('td').find_next_sibling('td')
            captions = [c.strip() for c in captions.text.splitlines() if len(c) > 0]
            if len(captions) > 0:
                annot[img] = captions

        print('{} image/caption pairs'.format(len(annot)))
        json.dump(annot, open(annotations, 'w'))
        print('{} saved'.format(annotations))
    else:
        annot = json.load(open(annotations, 'r'))
        print('{} already exist'.format(annotations))

    # NLP/Tagging model
    nlp = spacy.load(settings.SPACY_MODEL)
    if args.mode == 'noun':
        extractor = NounTags(nlp, alpha=args.alpha, syntactic=args.syntactic)
    elif args.mode == 'compound':
        extractor = CompoundTags(nlp, alpha=args.alpha, syntactic=args.syntactic)
    else:
        raise RuntimeError('not a valid mode')

    # compute tags from captions
    if not exists(args.output):
        res = pascal_tags(extractor, annot)

        if args.unrelated:
            sample_unrelated_inplace(res, settings.N_UNRELATED)

        if not exists(dirname(args.output)):
            os.makedirs(dirname(args.output))
        json.dump(res, open(args.output, 'w'))
        print('{} saved'.format(args.output))
    else:
        print('{} already exist'.format(args.output))

    # generate HTML
    if args.html:
        data = json.load(open(args.output, 'r'))
        htmlfile = splitext(args.output)[0] + '.html'
        htmlize(data, htmlfile)
