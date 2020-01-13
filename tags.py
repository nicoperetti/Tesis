"""Tags extractor Class."""
# coding: utf-8

import numpy as np

import settings

import spacy

from collections import Counter, defaultdict

from utils import check_hyper


class BaseEstimator(object):
    """ Base class for tag estimation
    Args:
       nlp (spacy model): NPL model
    """
    def __init__(self, nlp=None, alpha=0.5):
        if nlp is None:
            self.nlp = spacy.load(settings.SPACY_MODEL)
        else:
            self.nlp = nlp
            if not isinstance(self.nlp, spacy.language.Language):
                raise RuntimeError('not a valid model')

        self.alpha = float(alpha)
        if alpha > 1. or alpha < 0.:
            raise ValueError('alpha must be in [0, 1]')

    def process(self, caption_list):
        """
        Args:
           caption_list (list): captions (strings)

        Returns:
           tags, scores: list of ranked tags and their scores
        """
        captions = [self.nlp(c) for c in caption_list]

        tags, scores = self._process(captions)

        # sort by decreasing score
        idxs = np.argsort(scores)[::-1]

        return ([tags[i] for i in idxs],
                [scores[i] for i in idxs])

    def _process(self, captions):
        """
        Args:
           captions (list): captions (spacy docs)

        Returns:
           tags, scores: list tags and their scores
        """
        raise NotImplementedError('do what?')

    def r(self, counts, locs):
        # tag scores
        scores = self.alpha * np.array(counts) / sum(counts) + \
            (1 - self.alpha) * (1 - np.array(locs))

        return scores


class CountLocTags(BaseEstimator):
    """Base class for tag estimation using tag counts and relative locations
    Args:
       nlp (spacy model): NPL model

       alpha (float): the score for a tag W is computed as:

            alpha * relative_frequency(W) + (1-alpha) *(1-relative_location(W))

            the use of 1-relative_location is to give more importance to the
            words that appear first in the sentence.

       min_count (int): filter-out words with less than min_count occurrences
    """
    def __init__(self, nlp=None, alpha=0.5, min_count=0):
        super().__init__(nlp, alpha)

        self.min_count = int(min_count)

    def preprocess(self, caption_list):
        """
        Args:
           caption_list (list): captions (strings)

        Returns:
           tags, scores: list of ranked tags and their scores
        """
        captions = [self.nlp(c) for c in caption_list]

        # compute tags and relative locations
        tokens, rlocs = self._process(captions)

        # unique tags and their counts
        unique = dict(Counter(tokens))

        # list of tags
        tags = list(unique.keys())

        # counts
        counts = [unique[w] for w in tags]

        # filter-out tokens with a count lower thatn min_count
        tags = [w for i, w in enumerate(tags) if counts[i] >= self.min_count]
        counts = [c for c in counts if c >= self.min_count]

        # relative location within the sentences
        locs = []
        for w in tags:
            w_locs = [rlocs[i] for i, w_ in enumerate(tokens) if w_ == w]
            locs.append(min(w_locs))

        # sort by decreasing score
        idxs = np.argsort(locs)

        return ([tags[i] for i in idxs],
                [counts[i] for i in idxs],
                [locs[i] for i in idxs])

    def process(self, caption_list):

        tags, counts, rlocs = self.preprocess(caption_list)

        scores = list(self.r(counts, rlocs))
        idxs = np.argsort(scores)[::-1]
        return ([tags[i] for i in idxs],
                [scores[i] for i in idxs])


class NounTags(CountLocTags):
    """ Tag (nouns) estimation based on counts and relative locations.
    Args:
       nlp (spacy model): NPL model

       alpha (float): the score for a tag W is computed as:

          alpha * relative_frequency(W) + (1-alpha) * (1-relative_location(W))

          the use of 1-relative_location is to give more importance to the
          words that appear first in the sentence, or upper in the syntactic
          tree (if syntactic=True).

       min_count (int): filter-out words with less than min_count occurrences

       has_vector (bool): if True, consider only tags with a word embedding
    """
    def __init__(self, nlp=None, alpha=0.5, min_count=0, has_vector=True, syntactic=False):
        super().__init__(nlp, alpha, min_count)

        self.has_vector = bool(has_vector)

        self.syntactic = bool(syntactic)

    def _process(self, captions):
        if self.has_vector:
            is_ok = lambda w: bool(w.has_vector and w.pos_ == 'NOUN')
        else:
            is_ok = lambda w: bool(w.pos_ == 'NOUN')

        # list of (lemmatized_noun, relative_location_within_the_sentence)pairs
        res = []
        for c in captions:
            if self.syntactic:
                rlocs = [len(list(w.ancestors)) for w in c]
                norm = float(max(rlocs))
            else:
                rlocs = range(len(c))
                norm = float(len(c))
            res += [(w.lemma_, i / norm) for i, w in zip(rlocs, c) if is_ok(w)]
        tokens, rlocs = zip(*res)

        return tokens, rlocs


class CompoundTags(CountLocTags):
    """ Tag (compound nouns) estimation based on counts and relative locations.
    Args:
       nlp (spacy model): NPL model

       alpha (float): the score for a tag W is computed as:

          alpha * relative_frequency(W) + (1-alpha) * (1-relative_location(W))

          the use of 1-relative_location is to give more importance to the
          words that appear first in the sentence, or upper in the syntactic
          tree (if syntactic=True).

       min_count (int): filter-out words with less than min_count occurrences
    """
    def __init__(self, nlp=None, alpha=0.5, min_count=0, syntactic=False):
        super().__init__(nlp, alpha, min_count)

        self.syntactic = bool(syntactic)

    def _process(self, captions):
        res = []
        for c in captions:
            if self.syntactic:
                rlocs = [len(list(w.ancestors)) for w in c]
                norm = float(max(rlocs))
            else:
                rlocs = range(len(c))
                norm = float(len(c))
            for i, (w, rloc) in enumerate(zip(c, rlocs)):
                if w.pos_ == 'NOUN':
                    if c[i - 1].dep_ == 'compound':
                        continue
                    if w.dep_ == 'compound':
                        w = '{} {}'.format(c[i].lemma_, c[i + 1].lemma_)
                    else:
                        w = w.lemma_
                    res.append((w, rloc / norm))
        tokens, rlocs = zip(*res)

        return tokens, rlocs


class SyntacticMergedTags(BaseEstimator):
    """Syntactic Merged Tag estimation based on counts and relative locations."""

    def __init__(self, other, hyperonymcheck=False):
        """Class initializer.

        Args:
            other(class): Primitive tag extractor. Could be Noun or Compound Nouns
        """
        super().__init__(other.nlp, other.alpha)
        self.other = other
        self.hyperonymCheck = hyperonymcheck

    def process(self, caption_list):
        """Retrive tags from captions.

        Args:
           caption_list (list): captions (strings)

        Returns:
           tags, scores: list of ranked tags and their scores
        """
        tags, counts, rlocs = self.other.preprocess(caption_list)

        captions = [self.nlp(c) for c in caption_list]

        # build a token dict indexed by lemma and POS tag (for noun and verbs)
        token_dict = defaultdict(set)
        for sent in captions:
            for token in sent:
                if token.pos_ in ['NOUN', 'VERB']:
                    token_dict[token.lemma_, token.pos_].add(token)
        token_dict = dict(token_dict)

        # keep only repeated tokens
        for k in list(token_dict):
            if len(token_dict[k]) == 1:
                del token_dict[k]

        # build children token dict indexed by syntactic function
        child_dict = defaultdict(set)
        for k, tokens in token_dict.items():
            for token in tokens:
                for ctoken in token.children:
                    if ctoken.pos_ in ['NOUN', 'VERB']:
                        child_dict[k, ctoken.dep_].add(ctoken)
        child_dict = dict(child_dict)

        # keep only repeated children
        for k in list(child_dict):
            if len(child_dict[k]) == 1:
                del child_dict[k]

        # merge tags keeping the highest scored
        remove_idxs = set()
        for tokens in child_dict.values():
            idxs = set(tags.index(t.lemma_) for t in tokens if t.lemma_ in tags)
            idxs = sorted(idxs)

            # Hyperonym checker
            if self.hyperonymCheck:
                tags_to_check = [tags[i] for i in idxs]
                if check_hyper(tags_to_check):
                    remove_idxs.update(idxs[1:])
                    for id_ in idxs[1:]:
                        counts[idxs[0]] += counts[id_]
            else:
                remove_idxs.update(idxs[1:])
                for id_ in idxs[1:]:
                    counts[idxs[0]] += counts[id_]

        new_tags = [t for i, t in enumerate(tags) if i not in remove_idxs]
        new_rlocs = [r for i, r in enumerate(rlocs) if i not in remove_idxs]
        new_counts = [c for i, c in enumerate(counts) if i not in remove_idxs]

        new_scores = list(self.r(new_counts, new_rlocs))

        # sorting the tags and its scores in decrecing order
        idxs = np.argsort(new_scores)[::-1]

        return ([new_tags[i] for i in idxs],
                [new_scores[i] for i in idxs])
