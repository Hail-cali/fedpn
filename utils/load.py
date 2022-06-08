import json
import cv2
import numpy as np
import os
import pickle
import torch
from typing import Optional, Callable, AnyStr, Any
from torchvision.datasets import CIFAR10
import torchvision.transforms as TF


class CIFAR10Dataset(CIFAR10):
    base_folder = ''

    def __init__(
            self,
            root: str,
            image_set: str = 'train',

            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            cat_type: Optional[AnyStr] = None,
    ) -> None:

        transform = TF.Compose(
            [TF.ToTensor(),
             TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])

        super(CIFAR10Dataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        if image_set in ['train', 'global']:
            self.train = True
        else:
            self.train = False
        # self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays

        cat_list, batch_num = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '1')
        image_dir = os.path.join(self.root, self.base_folder)

        if cat_type:

            mapper = {'client_vehicle': ([0,1,8,9],'2'),
                      'client_animal': ([2,3,4,5,6,7],'3'),
                      'client_ground': ([1,3,4,5,6,9],'4'),
                      'client_without_dog_cat_bird':([0,1,4,6,7,8,9],'5'),
                      'client_not_ground': ([0,2,8],'5'),

                        # all class for test, deploy
                      'client_all': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '4'),
                      'global': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '1'),
                      'server': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '3'),

                        # except one class
                      'client_without_airplane': ([1, 2, 3, 4, 5, 6, 7, 8, 9], '2'),
                      'client_without_cat': ([0, 1, 2, 4, 5, 6, 7, 8, 9], '3'),
                      'client_without_dog': ([0, 1, 2, 3, 4, 6, 7, 8, 9], '4'),

                      'train_pretrained': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '5'),

                      }
            cat_list, batch_num = mapper[cat_type]

        if image_set == 'val':
            file_path = os.path.join(image_dir, 'test_batch')
        else:
            file_path = os.path.join(image_dir, 'data_batch_' + batch_num)

        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

            for image, target in zip(entry['data'], entry['labels']):
                if self.train and target in cat_list :
                    self.targets.append(target)
                    self.data.append(image)

                elif not self.train:
                    self.targets.append(target)
                    self.data.append(image)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # self.targets = np.hstack(self.targets)
        self.targets = torch.LongTensor(self.targets)
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.meta["filename"])

        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


class Config:
    def __init__(self, json_path):

        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class LoadImage(object):

    def __init__(self, space='BGR'):
        self.space = space

    def __call__(self, path_img):
        return cv2.imread(path_img)


