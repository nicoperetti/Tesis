#coding: utf-8

from PIL import Image

import numpy as np
np.seterr(all='raise')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from scipy.spatial.distance import cdist

GPU_AVAILABLE = torch.cuda.is_available()


class FeatureExtractor(object):
    def __init__(self, flip=False):
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.flip = bool(flip)
        self.net = None

    def process(self, x):
        x = self.transforms(x)
        if GPU_AVAILABLE:
            x = x.cuda()

        x = Variable(x.unsqueeze_(0))
        f = self.net(x)
        if self.flip:
            raise NotImplementedError('TODO: left-right flip tensor')

        f = F.normalize(f, p=2, dim=1)
        if GPU_AVAILABLE:
            f = f.cpu()
        return f.data.numpy().squeeze()

    # def forward(self, x):
    #     x = self.transforms(x)
    #     outputs = []
    #     for name, module in self.submodule._modules.items():
    #         x = module(x)
    #         if name in self.extracted_layers:
    #             outputs += [x]
    #     return outputs + [x]


# ------------------------------------------------------------------------------


class VGGFeatureExtractor(FeatureExtractor):
    def __init__(self, model, flip=False):
        super().__init__(flip)
        for p in list(model.features.parameters()) + \
            list(model.classifier.parameters()):
            p.requires_grad = False
        #features = list(model.features.children())
        classifier = list(model.classifier.children())[:-2]
        #self.net = nn.Sequential(*(features + classifier))
        model.classifier = nn.Sequential(*classifier)
        self.net = model
        if GPU_AVAILABLE:
            self.net.cuda()


class VGG16FeatureExtractor(VGGFeatureExtractor):
    def __init__(self, flip=False):
        super().__init__(models.vgg16(pretrained=True), flip)


class VGG19FeatureExtractor(VGGFeatureExtractor):
    def __init__(self, flip=False):
        super().__init__(models.vgg19(pretrained=True), flip)


# ------------------------------------------------------------------------------


class ResNetFeatureExtractor(FeatureExtractor):
    def __init__(self, model, flip=False):
        super().__init__(flip)
        for p in list(model.parameters()):
            p.requires_grad = False
        features = list(model.children())[:-1]
        self.net = nn.Sequential(*features)
        if GPU_AVAILABLE:
            self.net.cuda()


class ResNet50FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, flip=False):
        super().__init__(models.resnet50(pretrained=True), flip)


class ResNet101FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, flip=False):
        super().__init__(models.resnet101(pretrained=True), flip)


class ResNet152FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, flip=False):
        super().__init__(models.resnet152(pretrained=True), flip)


# ------------------------------------------------------------------------------


class RMACFeatureExtractor(object):
    'simplified R-MAC feature extractor'

    def __init__(self, input_size=512, scales=[1,2,4], flip=False):

        model = models.vgg16(pretrained=True)
        if GPU_AVAILABLE:
            model.cuda()
        for p in list(model.features.parameters()):
            p.requires_grad = False
        features = list(model.features.children())[:-1]
        self.net = nn.Sequential(*features)

        self.input_size = int(input_size)
        self.scales = list(scales)
        self.flip = bool(flip)

        self._transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

        # self.pca = None

    def _region_level_features(self, f):
        N, C, H, W = f.size()

        ks = min(H, W)

        feat = []

        for s in self.scales:
            kernel_size = int(round(ks / s))
            if kernel_size <= 1:
                break
            stride = max(1, kernel_size // 2)  # overlap ~50%
            feat_ = F.max_pool2d(f, kernel_size=kernel_size, stride=stride).data
            _, _, h, w = feat_.size()
            feat_.resize_((N, C, h*w))
            feat_.transpose_(1, 2)
            feat_.squeeze_(0)

            feat.append(feat_)

        feat = torch.cat(feat, 0)
        return F.normalize(feat, p=2, dim=1)

    def _process(self, img):
        img = self._transform(img)
        if img.dim() == 3:
            img.unsqueeze_(0)

        img = Variable(img)
        if GPU_AVAILABLE:
            img = img.cuda()

        # activation map volume (tensor)
        f = self.net(img)

        # multi-scale region-level feature pooling
        f = self._region_level_features(f)

        # global feature pooling
        f = torch.sum(f, 0)

        return f

        # if self.pca is not None:
        #     feat = self.pca.transform(feat)
        #     feat = preprocessing.normalize(feat, norm='l2')
        # feat = np.sum(feat, axis=0).reshape(1, -1)
        # return preprocessing.normalize(feat, norm='l2').squeeze()

    def process(self, img):
        f = self._process(img)
        if self.flip:
            f += self._process(img.transpose(Image.FLIP_LEFT_RIGHT))

        f = F.normalize(f.unsqueeze_(0), p=2, dim=1).squeeze_(0)

        if GPU_AVAILABLE:
            f = f.cpu()

        return f.numpy()
