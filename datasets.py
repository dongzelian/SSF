# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform



import hub

# from .nabirds import NABirdsDataset
# from .stanford_dog import dogs
# from .cub2011 import Cub2011


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp



def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH, train=is_train, transform = transform, download=True)
        nb_classes = 100
    elif config.DATA.DATASET == 'DTD':
        prefix = 'train' if is_train else 'val' #'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 47
    elif config.DATA.DATASET == 'oxford_flowers':
        prefix = 'train' if is_train else 'test'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    elif config.DATA.DATASET == 'euro_sat':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif config.DATA.DATASET == 'pcam':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    elif config.DATA.DATASET == 'pets':
        prefix = 'train' if is_train else 'test'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 37
    elif config.DATA.DATASET == 'snorb_azimuth':
        prefix = 'train' if is_train else 'test'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 18
    elif config.DATA.DATASET == 'snorb_elevation':
        prefix = 'train' if is_train else 'test'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 9
    elif config.DATA.DATASET == 'stanford_cars':
        prefix = 'train' if is_train else 'test'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 196
    # elif config.DATA.DATASET == 'stanford_dogs':
    #     prefix = 'train' if is_train else 'test'
    #     dataset = dogs(root=config.DATA.DATA_PATH, train=is_train, transform=transform)
    #     #root = os.path.join(config.DATA.DATA_PATH, prefix)
    #     #dataset = datasets.ImageFolder(config.DATA.DATA_PATH, transform=transform)
    #     nb_classes = 120
    # elif config.DATA.DATASET == 'nabirds':
    #     prefix = 'train' if is_train else 'val'
    #     #root = os.path.join(config.DATA.DATA_PATH, prefix)
    #     #dataset = datasets.ImageFolder(root, transform=transform)
    #     ds = hub.dataset('hub://activeloop/nabirds-dataset-' + prefix) # Hub Dataset
    #     dataset = NABirdsDataset(ds=ds, transform=transform)
    #     nb_classes = 55
    # elif config.DATA.DATASET == 'cub2011':
    #     prefix = 'train' if is_train else 'test'
    #     dataset = Cub2011(root=config.DATA.DATA_PATH, train=is_train, transform=transform)
    #     #root = os.path.join(config.DATA.DATA_PATH, prefix)
    #     #dataset = datasets.ImageFolder(config.DATA.DATA_PATH, transform=transform)
    #     nb_classes = 200

    elif config.DATA.DATASET == 'svhn':
        prefix = 'train' if is_train else 'test'
        dataset = datasets.SVHN(root=config.DATA.DATA_PATH, split=prefix, transform = transform, download=True)
        nb_classes = 10
    elif config.DATA.DATASET == 'dsprites_loc':
        prefix = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(root=config.DATA.DATA_PATH, transform = transform)
        nb_classes = 16
    elif config.DATA.DATASET == 'dsprites_orient':
        prefix = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(root=config.DATA.DATA_PATH, split=prefix, transform = transform, download=True)
        nb_classes = 16





    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if config.DATA.DATASET == 'cifar100': #'imagenet':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )

            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
            return transform

        elif config.DATA.DATASET == 'imagenet':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )

            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
            return transform


        #TODO: my data augmentation
        elif config.DATA.DATASET == 'oxford_flowers':
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'stanford_cars':
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans
        elif config.DATA.DATASET == 'stanford_dogs':
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'nabirds':
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans
        elif config.DATA.DATASET == 'cub2011':
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans




        elif config.DATA.DATASET == 'pets':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans
        
        elif config.DATA.DATASET == 'euro_sat':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'pcam':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'snorb_azimuth':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'snorb_elevation':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'svhn':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans
        elif config.DATA.DATASET == 'dsprites_loc':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans
        elif config.DATA.DATASET == 'dsprites_orient':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans

        elif config.DATA.DATASET == 'DTD':
            trans = transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

            return trans



    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
