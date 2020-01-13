#coding: utf-8

import json

from os.path import join

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from PIL import Image

from utils import normalize_rows

CUDA_AVAILABLE = torch.cuda.is_available()



class PairedTensorDataset(data.TensorDataset):
    """
    Args:
       input_tensor (list, Tensor): input modality
       output_tensor (list, Tensor): output modality
    """
    def __init__(self, input_tensor, output_tensor):
        super().__init__()

        if isinstance(input_tensor, (list, tuple)):
            n_input = len(input_tensor)
        else:
            n_input = input_tensor.size(0)

        if isinstance(output_tensor, (list, tuple)):
            n_output = len(output_tensor)
        else:
            n_output = output_tensor.size(0)

        if n_input != n_output:
            raise ValueError('input and output must have the same number of elements')
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.output_tensor[idx]

    def __len__(self):
        return self.input_tensor.size(0)

    # @staticmethod
    # def collate_fn(batch):
    #     if isinstance(batch, (list, tuple)):
    #         return batch
    #     return default_collate(batch)

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/issues/1512
        # Note that batch is a list
        batch = list(map(list, zip(*batch)))  # transpose list of list
        out = None
        # You should know that batch[0] is a fixed-size tensor since you're using your customized Dataset
        # reshape batch[0] as (N, H, W)
        # batch[1] contains tensors of different sizes; just let it be a list.
        # If your num_workers in DataLoader is bigger than 0
        #     numel = sum([x.numel() for x in batch[0]])
        #     storage = batch[0][0].storage()._new_shared(numel)
        #     out = batch[0][0].new(storage)
        batch[0] = torch.stack(batch[0], 0, out=out)
        return batch



# class TaggedFeatures(data.TensorDataset):
#     """
#     Args:
#        features_path (str): (base) path to the features
#        tags_file (str): .json file with ranked tags (see format at coco.py)
#        set_ (str): data subset (a key on the first level of the .json file)
#     """
#     def __init__(self, features_path, tags_file):
#         super().__init__()

#         path = abspath(path)
#         if not exists(path):
#             raise OSError('{} does not exists'.format(path))
#         flist, fpath = get_file_list(path, ('.dat',))

#         assert data_tensor.size(0) == len(target_tensor)
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor

#     def __getitem__(self, idx):
#         return self.data_tensor[idx], self.target_tensor[idx]


class TaggedImages(data.Dataset):
    """
    Args:
       path (str): (base) path to the dataset
       tags_file (str): .json file with ranked tags (see format at coco.py)
       set_ (str): data subset (a key on the first level of the .json file)
    """
    def __init__(self, path, tags_file, set_):
        super().__init__()

        self._path = abspath(path)
        if not exists(self._path):
            raise OSError('{} doesn\'t exists'.format(self._path))

        self._data = json.load(open(tags_file, 'r'))

        self._set = set_
        if not self._set in data.keys():
            raise ValueError('{} isn\'t a valid key'.format(set_))

        # sort images for reproducibility reasons
        sorted_keys = sorted(list(self._data[self._set]))
        self._data[self._set] = [self._data[self._set][k] for k in sorted_keys]

        # L2 normalization of word embeddings
        if not 'vectors' in self._data:
            raise RuntimeError('there are no word embeddings for this dataset')
        vectors = self._data['vectors']
        for tag, vec in vectors.items():
            if vec is not None:
                vectors[tag] = normalize_rows(vec).astype(np.float32)

    # def tag_embeddings(self, tags):
    #     emb = [self._data['vectors'][w] for w in tags]
    #     return normalize_rows(emb, )

    def __getitem__(self, idx):
        data = self._data[self._set]
        imfile = data[idx]['file_name']
        im = Image.open(join(self._path, imfile)).convert('RGB')

        tags = data[idx]['tags']

        # # scores = data[idx]['scores']
        # counts = data[idx]['counts']
        # offsets = [np.median(off) for off in data[idx]['offsets']]
        # alpha = 0.9
        # scr = alpha * np.array(counts) + (1-alpha) * (1-np.array(offsets))
        # tags = [tags[i] for i in np.argsort(scr)[::-1]]

        emb = np.array([self._data['vectors'][w] for w in tags])
        return im, torch.from_numpy(emb)

    def __len__(self):
        return len(self._data[self._coco_set])
