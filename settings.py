from os.path import expanduser, join

# ------------------------------------------------------------------------------
# tags from captions

# spacy language model
SPACY_MODEL = "en_core_web_md"

# datasets
COCO_PATH = "./COCO"
PASCAL_PATH = "./pascal-sentences"
MIRFLICKR_PATH = "./MIRFlickr"

# word2vec dict
WORD2VEC_MODEL = "./GoogleNews-vectors-negative300.bin"

# fastText dict
FASTTEXT_MODEL = "./wiki-news-300d-1M.vec"

# number of unrelated tags per sample
N_UNRELATED = 100


BERT_VOCAB = "bert/uncased_L-12_H-768_A-12/vocab.txt"
BERT_CONFIG = "bert/uncased_L-12_H-768_A-12/bert_config.json"
BERT_CHK = "bert/uncased_L-12_H-768_A-12/bert_model.ckpt"
