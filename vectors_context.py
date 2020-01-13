"""Word Embedding Script."""
# coding: utf-8

from os.path import exists, splitext

import argparse

import numpy as np

import json

import spacy

import settings

from utils import progressbar

from bertEmb import BertModel

import modeling

import tensorflow as tf

import tokenization

from tqdm import tqdm

from collections import defaultdict


is_ok = lambda w: bool(w.pos_ == 'NOUN')


def postprocessing(orig_tokens_list):
    res = []
    for orig_tokens in orig_tokens_list:
        orig_to_tok_map = []
        bert_tokens = []

        tokenizer = tokenization.FullTokenizer(
            vocab_file=settings.BERT_VOCAB, do_lower_case=True)

        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(tokenizer.tokenize(orig_token.text))
        bert_tokens.append("[SEP]")
        res.append(orig_to_tok_map)
    return res


def bert_mapping(prediction_list, post_processing_list, sets_):
    res = []
    assert len(prediction_list) == len(post_processing_list) == len(sets_)
    for prediction, mapping, origin_tok in zip(prediction_list, post_processing_list, sets_):
        emb = [w for i, (_,w) in enumerate(prediction) if i in mapping]
        res.append([(tok, emb[i]) for i, tok in enumerate(origin_tok) if is_ok(tok)])
        assert len(emb) == len(origin_tok)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute word embeddings from tag list',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda s: s.lower() in ("yes", "true", "t", "1", "on"))
    parser.add_argument('tags', help='json file. It must have a \'tags\' key with a list of tags', type=str)
    args = parser.parse_args()

    try:
        tags_file = json.load(open(args.tags, 'r'))
        tags = tags_file['tags']
    except:
        raise IOError("something is wrong with the tags file")

    bert_file = splitext(args.tags)[0] + '.cbert'

    nlp = None
    tokens = []
    if not (exists(bert_file)):
        nlp = spacy.load(settings.SPACY_MODEL)

        # token = list of lemmatized words (contemplates the case of compound tags
        # or tags that has not been properly lemmatized)
        print('precomputing token list ...')
        for w0 in progressbar(tags):
            w1 = [w.lemma_ for w in nlp(w0)]
            tokens.append(w1)

    # Bert
    if not exists(bert_file):
        if not exists(settings.BERT_CONFIG) and \
           not exists(settings.BERT_CHK)and \
           not exists(settings.BERT_VOCAB):
            raise IOError("bert model does\'t not exists.")

        batch_size = 16
        bert_model = BertModel(bert_config_file=settings.BERT_CONFIG,
                               init_checkpoint=settings.BERT_CHK,
                               vocab_file=settings.BERT_VOCAB, batch_size=batch_size)

        print('precomputing Bert vectors ...')
        d = defaultdict(list)
        bert = defaultdict(list)
        # bert = dict((w, None) for w in tags)
        for mode in ['val2014', 'train2014']:
            range(0, len(tokens), batch_size)
            keys = list(tags_file[mode].keys())
            n = 3
            for ii in tqdm(range(0, len(keys), n), total=len(keys)//n):
                captions_list = []
                for kk in keys[ii:ii+n]:
                    caps = tags_file[mode][kk]["captions"]
                    caps = [cap.strip().replace("  ", " ") for cap in caps]
                    captions_list += caps
                sents_list = [nlp(c) for c in captions_list]

                prediction_list = bert_model.predict(captions_list)
                post_processing_list = postprocessing(sents_list)
                try:
                    result = bert_mapping(prediction_list, post_processing_list, sents_list)
                    for p in result:
                        for i, j in p:
                            d[i.lemma_].append(j)
                except:
                    print(keys[ii:ii+n])
                    continue

        d = dict(d)
        for tok, vec_list in d.items():
            bert[tok] = list(np.average(np.array(vec_list), axis=0))

        bert = dict(bert)
        json.dump(bert, open(bert_file, 'w'))
        print("{} saved".format(bert_file))

        del bert_model, bert
    else:
        print("{} already exists".format(bert_file))
