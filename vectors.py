"""Word Embedding Script."""
# coding: utf-8

from os.path import exists, splitext

import argparse

import numpy as np

import json

import spacy

from gensim.models import KeyedVectors

import settings

from utils import progressbar

from bertEmb import BertModel

import io


def load_vectors(fname):
    """Load Vectors."""
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ', maxsplit=1)
        data[tokens[0]] = np.fromstring(tokens[1], dtype=np.float32, sep=' ')
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute word embeddings from tag list',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda s: s.lower() in ("yes", "true", "t", "1", "on"))
    parser.add_argument('tags', help='json file. It must have a \'tags\' key with a list of tags', type=str)
    args = parser.parse_args()

    try:
        tags = json.load(open(args.tags, 'r'))['tags']
    except:
        raise IOError("something is wrong with the tags file")

    glove_file = splitext(args.tags)[0] + '.glove'

    word2vec_file = splitext(args.tags)[0] + '.word2vec'

    fasttext_file = splitext(args.tags)[0] + '.fasttext'

    bert_file = splitext(args.tags)[0] + '.bert'

    nlp = None
    tokens = []
    if not (exists(glove_file) and exists(word2vec_file) and exists(fasttext_file) and exists(bert_file)):
        nlp = spacy.load(settings.SPACY_MODEL)

        # token = list of lemmatized words (contemplates the case of compound tags
        # or tags that has not been properly lemmatized)
        print('precomputing token list ...')
        for w0 in progressbar(tags):
            w1 = [w.lemma_ for w in nlp(w0)]
            tokens.append(w1)

    # spacy GloVe
    if not exists(glove_file):
        print('precomputing GloVe vectors ...')
        glove = dict((w, None) for w in tags)
        for i, tok in progressbar(enumerate(tokens), total=len(tokens)):
            vecs = [w.vector for w in nlp(' '.join(tok)) if w.has_vector]
            if len(vecs) > 0:
                glove[tags[i]] = np.mean(np.atleast_2d(vecs), axis=0).tolist()

        json.dump(glove, open(glove_file, 'w'))
        print("{} saved".format(glove_file))

        del glove
    else:
        print("{} already exists".format(glove_file))

    # gensim word2vec
    if not exists(word2vec_file):
        if not exists(settings.WORD2VEC_MODEL):
            raise IOError("word2vec model does\'t not exists. You can download"
                          "it from https://code.google.com/archive/p/word2vec/"
                          "and set the WORD2VEC_MODEL variable in settings.py "
                          "accordingly")

        w2v = KeyedVectors.load_word2vec_format(settings.WORD2VEC_MODEL, binary=True)

        print('precomputing word2vec vectors ...')
        word2vec = dict((w, None) for w in tags)
        for i, tok in progressbar(enumerate(tokens), total=len(tokens)):
            vecs = [w2v[w] for w in tok if w in w2v]
            if len(vecs) > 0:
                word2vec[tags[i]] = np.mean(np.atleast_2d(vecs), axis=0).tolist()

        json.dump(word2vec, open(word2vec_file, 'w'))
        print("{} saved".format(word2vec_file))

        del w2v, word2vec
    else:
        print("{} already exists".format(word2vec_file))

    # fastText
    if not exists(fasttext_file):
        if not exists(settings.FASTTEXT_MODEL):
            raise IOError("fastText model does\'t not exists. You can download"
                          "it from https://fasttext.cc/docs/en/english-vectors"
                          "and set the FASTTEXT_MODEL variable in settings.py "
                          "accordingly")

        ft = load_vectors(settings.FASTTEXT_MODEL)

        print('precomputing fastText vectors ...')
        fasttext = dict((w, None) for w in tags)
        for i, tok in progressbar(enumerate(tokens), total=len(tokens)):
            vecs = [ft[w] for w in tok if w in ft]
            if len(vecs) > 0:
                fasttext[tags[i]] = np.mean(np.atleast_2d(vecs), axis=0).tolist()

        json.dump(fasttext, open(fasttext_file, 'w'))
        print("{} saved".format(fasttext_file))

        del ft, fasttext
    else:
        print("{} already exists".format(fasttext_file))

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
        bert = dict((w, None) for w in tags)
        from tqdm import tqdm
        for i in tqdm(range(0, len(tokens), batch_size), total=len(tokens)//batch_size):
            toks = tokens[i:i + batch_size]
            toks = [tok[0] for tok in toks]
            preds = bert_model.predict(toks)
            vecs = [pred[1][1] for pred in preds]
            for j, vec in enumerate(vecs):
                bert[tags[i + j]] = vec

        json.dump(bert, open(bert_file, 'w'))
        print("{} saved".format(bert_file))

        del bert_model, bert
    else:
        print("{} already exists".format(fasttext_file))
