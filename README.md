Generar tags desde captions de MS-COCO
=======================================

1. Instalar tools

```sh
$ pip3 install --user pycocotools
$ pip3 install --user spacy
$ pip3 install --user gensim
```

1. Instalar modelo spacy con embeggins preentrenados

$ python3 -m spacy download en_core_web_md

1. Desgargar el archivo GoogleNews-vectors-negative300.bin.gz de
https://code.google.com/archive/p/word2vec/ y descomprimilo en el directorio del
proyecto (o crear symlink).

1. Descargar el archivo wiki-news-300d-1M.vec.zip de
https://fasttext.cc/docs/en/english-vectors y descomprimilo en el directorio del
proyecto (o crear symlink).

1. Descargar anotaciones de MS-COCO.

```sh
$ mkdir -p COCO; cd COCO
$ wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
$ unzip annotations_trainval2014.zip
$ cd ..
```

1. Editar settings.py, setear variables y correr:

```sh
$ python3 coco.py --mode=noun --output=data/coco_noun.tags
```

eso genera un json el que tiene un diccionation con campos 'train2014',
'val2014' y 'tags'.  Los primeros dos son diccionarios indexados por ID. Cada
entrada corresponde a una imagen. El último corresponde a una lista con el
conjunto completo de tags. Por ejemplo, para ver los tags asociados a la imagen
de ID '57870', ordenados por score (¿?), hacer:

```python
import json
data = json.load(open('data/coco_nouns.tags'))
print(data['train2014']['57870']['tags'])
```

Al generar los tags, se puede pasar la opción --html. Esto genera un html con
las imágenes, captions, tags y scores asociados. Para eso hay que haber
descargado las imágenes de coco y haberlas descomprimido en directorios
\<COCO_PATH\>/train2014 y \<COCO_PATH\>/val2014 respectivamente (~20GB)

```sh
$ cd <COCO_PATH>
$ wget http://images.cocodataset.org/zips/train2014.zip
$ unzip train2014.zip
$ wget http://images.cocodataset.org/zips/val2014.zip
$ unzip val2014.zip
```

En jupiterace, el dataset está en /home/jsanchez/resources/dataset/COCO

Para trabajar sobre la generación de tags, hacerlo sobre tags.py

1. Computar word embeddings sobre los tags generados

```sh
$ python3 vectors.py data/coco_noun.tags
```

Esto genera un par de archivos (.glove, .word2vec) en el mismo path del archivo
de tags. Cada uno corresponde a un json que almacena un diccionario (indexado
por tags individuales) de word embeddings.

Para obtener el word embedding de un tag en particular, hacer:

```python
import json
tags = json.load(open('data/coco_nouns.tags'))
glove = json.load(open('./data/coco_noun.glove', 'r'))
tag_list = tags['train2014']['57870']['tags']
vectors = [np.array(glove[w]) for w in tag_list]
```

Computar features visuales
==========================

```sh
$ python3 features.py <DATASET_PATH> --output_path=./data/ --features=vgg16
```

Se crea un .dat por imagen, manteniéndose la estructura de directorios del
dataset.


Entrenar modelo bilineal
========================

```sh
$ python3 train_bilinear.py <TAGS_FILE> <VECTORS_FILE> <FEATURES_PATH> --batch_size=16 --learning_rate=1e-4
```

Se puede usar la opción --debug al lanzar el entrenamiento, en cuyo caso se
utilizan 1000 muestras. Esto sirve para debug y ajuste de hiperparámetros.

Para ver las curvas de entrenamiento, instalar tensorboardX con pip y correr en
el mismo path en donde se lanzó el entrenamiento:

```sh
$ tensorboard --logdir ./runs  # runs es el directorio en donde se guardan los logs
```

y abrir la dirección http://127.0.0.1:6006 en un navegador.


Correr pruebas sobre MIRFlick25k (BROKEN)
================================

1. Descargar dataset y anotaciones

```sh
$ mkdir MIRFlickr && cd MIRFlickr
$ wget http://press.liacs.nl/mirflickr/mirflickr25k.v2/mirflickr25k.zip
$ unzip mirflickr25k.zip -d mirflickr25k
$ wget http://press.liacs.nl/mirflickr/mirflickr25k.v2/mirflickr25k_annotations_v080.zip
$ unzip mirflickr25k_annotations_v080.zip -d annotations
```

1. Setear los paths en settings.py y [computar features visuales](#computar-features-visuales) sobre las imágenes de MIRFlickr.

1. [Entrenar modelo](#entrenar-modelo-bilineal) en COCO

1. Evaluar modelo en MIRFlickr25k, apuntando a un modelo preentrenado y al directorio de features de MIRFlickr

```sh
$ python3 mirflickr.py checkpoints/model_0020.pth data/mirflickr25k/vgg16/
```