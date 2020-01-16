Generate Tags form MS-COCO captions
=======================================

1. Install tools

```sh
$ pip3 install --user pycocotools
$ pip3 install --user spacy
$ pip3 install --user gensim
```

1. Install Spacy model with its pre-trained embeddings.

$ python3 -m spacy download en_core_web_md

1. Download the file: GoogleNews-vectors-negative300.bin.gz from
https://code.google.com/archive/p/word2vec/ and unzip it inside the project directory.

1. Download the file wiki-news-300d-1M.vec.zip from
https://fasttext.cc/docs/en/english-vectors and unzip it inside the project directory.

1. Download MS-COCO anotaciones.

```sh
$ mkdir -p COCO; cd COCO
$ wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
$ unzip annotations_trainval2014.zip
$ cd ..
```

1. Edit settings.py with the correct path to the files.:

```sh
$ python3 coco.py --mode=noun --output=data/coco_noun.tags
```

The output will be a dictionary with the keys 'train2014', 'val2014' y 'tags'
the first two of them indexed by ID which is an unique image, the last key will be a
list of tags. In the following example we can see how to get tags asociated with the ID '57870'

```python
import json
data = json.load(open('data/coco_nouns.tags'))
print(data['train2014']['57870']['tags'])
```

```sh
$ cd <COCO_PATH>
$ wget http://images.cocodataset.org/zips/train2014.zip
$ unzip train2014.zip
$ wget http://images.cocodataset.org/zips/val2014.zip
$ unzip val2014.zip
```


In order to work wiht the tag generation we can do it over tags.py

1. Compute word embeddings over the generated tags.

```sh
$ python3 vectors.py data/coco_noun.tags
```

It will generate several files (.glove, .word2vec)

In order to get a word embedding of a particular tag we can do:


```python
import json
tags = json.load(open('data/coco_nouns.tags'))
glove = json.load(open('./data/coco_noun.glove', 'r'))
tag_list = tags['train2014']['57870']['tags']
vectors = [np.array(glove[w]) for w in tag_list]
```

Compute Visual Features
==========================

```sh
$ python3 features.py <DATASET_PATH> --output_path=./data/ --features=vgg19
```

It will create a .dat for each image.


Train a Bilinar Model
========================

```sh
$ python3 train_bilinear.py <TAGS_FILE> <VECTORS_FILE> <FEATURES_PATH> --batch_size=16 --learning_rate=1e-4
```

We can use a --debug option taking only 1000 samples for training.

In order to see the learning curves install tensorboardX with pip and run:


```sh
$ tensorboard --logdir ./runs  # runs in the same directory where the logs were saved
```

open: http://127.0.0.1:6006 in the browser.

Notebooks
========================
over notebook directory we have a couple of experiments such as:

- Fasttext for Tag generation: 001_Experiment_Fastext_for_tag_generation.ipynb

- Hyperonym: 002_experiment_merge_noun_hyper_filter.ipynb and Hyperonimos.ipynb

- Autoencoder: Autoencoder.ipynb

- Some ad-hoc models: pytorch.ipynb
