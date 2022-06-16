import json
import cv2
import numpy as np
import os
import pickle
import torch
from typing import Optional, Callable, AnyStr, Any
from torchvision.datasets import CIFAR10
import torchvision.transforms as TF
import numpy as np
from itertools import permutations, combinations
import random




class RANDOMCIFAR10Dataset(CIFAR10):

    base_folder = ''
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            image_set: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            cat_type: Optional[AnyStr] = None,
    ) -> None:

        # transform = TF.Compose(
        #     [TF.ToTensor(),
        #      TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #      ])

        transform = TF.Compose(
            [TF.ToTensor(),
             TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])

        super(RANDOMCIFAR10Dataset, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)

        if image_set in ['train', 'global']:
            self.train = True
        else:
            self.train = False

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        client = self.client(cat_type)
        print(f'{cat_type} :client info: ({client})')
        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')

                if cat_type in ['server', 'global']:
                    self.data.append(entry['data'])

                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

                else:
                    for image, target in zip(entry['data'], entry['labels']):
                        if target in client:
                            self.data.append(image)

                            if client[0] == target:
                                mapped_target = True
                            else:
                                mapped_target = False
                            self.targets.append(mapped_target)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        self.targets = torch.LongTensor(self.targets)

        self._load_meta()

    def client(self, cat_type):
        if cat_type in ['server', 'global']:
            self.cat_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            mapper ={'c1': [2, 7], 'c2': [3, 4], 'c3': [1, 2], 'c4': [0, 6], 'c5': [1, 9], 'c6': [3, 6], 'c7': [8, 9],
                     'c8': [5, 6], 'c9': [0, 7], 'c10': [1, 3], 'c11': [7, 8], 'c12': [7, 9], 'c13': [0, 1], 'c14': [2, 5],
                     'c15': [0, 2], 'c16': [1, 8], 'c17': [3, 8], 'c18': [3, 7], 'c19': [1, 4], 'c20': [1, 5], 'c21': [2, 8],
                     'c22': [4, 9], 'c23': [2, 4], 'c24': [4, 8], 'c25': [4, 7], 'c26': [6, 7], 'c27': [2, 9], 'c28': [0, 8],
                     'c29': [4, 5], 'c30': [0, 9], 'c31': [5, 9], 'c32': [3, 9], 'c33': [6, 8], 'c34': [4, 6], 'c35': [5, 8],
                     'c36': [6, 9], 'c37': [5, 7], 'c38': [1, 6], 'c39': [0, 5], 'c40': [2, 3], 'c41': [3, 5], 'c42': [2, 6],
                     'c43': [0, 3], 'c44': [1, 7], 'c45': [0, 4],  'c46': [1,9], 'c47': [2, 8], 'c48': [3, 8], 'c49': [4, 9],
                     'c50': [1, 5],'c51': [2, 7], 'c52': [3, 4], 'c53': [1, 2], 'c54': [0, 6], 'c55': [1, 9], 'c56': [3, 6], 'c57': [8, 9],
                     'c58': [5, 6], 'c59': [0, 7], 'c60': [1, 3], 'c61': [7, 8], 'c62': [7, 9], 'c63': [0, 1], 'c64': [2, 5],
                     'c65': [0, 2], 'c66': [1, 8], 'c67': [3, 8], 'c68': [3, 7], 'c69': [1, 4], 'c70': [1, 5], 'c71': [2, 8],
                     'c72': [4, 9], 'c73': [2, 4], 'c74': [4, 8], 'c75': [4, 7], 'c76': [6, 7], 'c77': [2, 9], 'c78': [0, 8],
                     'c79': [4, 5], 'c80': [0, 9], 'c81': [5, 9], 'c82': [3, 9], 'c83': [6, 8], 'c84': [4, 6], 'c85': [5, 8],
                     'c86': [6, 9], 'c87': [5, 7], 'c88': [1, 6], 'c89': [0, 5], 'c90': [2, 3], 'c91': [3, 5], 'c92': [2, 6],
                     'c93': [0, 3], 'c94': [1, 7], 'c95': [0, 4],  'c96': [1,9], 'c97': [2, 8], 'c98': [3, 8], 'c99': [4, 9],
                     'c100': [1, 5],
                     }

            return mapper[cat_type]

            # candi = list(combinations(range(10), 2))
            # random.shuffle(candi)
            # mapper = dict((f'c{i+1}',list(choose)) for i, choose in enumerate(candi))






    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.meta["filename"])

        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")

            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


class MappedCIFAR10Dataset(CIFAR10):
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

        # transform = TF.Compose(
        #     [TF.ToTensor(),
        #      TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #      ])

        transform = TF.Compose(
            [TF.ToTensor(),
             TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])

        super(MappedCIFAR10Dataset, self).__init__(root, transform=transform,
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
            mapper = {

                'c1': ([0, 9], '1'),
                'c2': ([1,8], '2'),
                'c3': ([2,7], '3'),
                'c4': ([3,6], '4'),
                'c5': ([4,5], '1'),
                'c6': ([0,8], '2'),
                'c7': ([1,7], '3'),
                'c8': ([2,6], '4'),
                'c9': ([3,5], '1'),
                'c10': ([4,9], '2'),

                'client_all': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '5'),
                'global': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '5'),
                'server': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], '5'),

            }



            cat_list, batch_num = mapper[cat_type]

        if image_set == 'val':
            file_path = os.path.join(image_dir, 'test_batch')

        else:
            file_path = os.path.join(image_dir, 'data_batch_' + batch_num)

        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

            for image, target in zip(entry['data'], entry['labels']):

                if cat_type in ['server', 'global']:
                    # skip = np.random.randint(8, 12)
                    # if (c % skip) == 0:
                    self.targets.append(target)
                    self.data.append(image)
                else:
                    #mapping
                    if target in cat_list:
                        if cat_list[0] == target:
                            mapped_target = True
                        else:
                            mapped_target = False
                        self.targets.append(mapped_target)
                        self.data.append(image)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

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


