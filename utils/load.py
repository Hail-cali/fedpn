import torch.utils.data as data
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def unpickle(file, batch_num):
    import pickle
    import os
    batch_file = 'data_batch_' + str(batch_num)
    file_path =  os.path.join(file, batch_file)
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

class ImageDataset(data.Dataset):

    def __init__(self, data, test_mode=False):
        super(ImageDataset, self).__init__()
        if not test_mode:
            self.X, self.y = self.make_dataset(data)
        else:
            self.X, self.y = self.make_dataset(data)
            self.X, self.y = self.X[:400], self.y[:400]
        # self.X_len = self.X.shape[0]


    def make_dataset(self, dict):
        return dict[b'data'].reshape(len(dict[b'data']), 3, 32, 32).astype('float32'), np.array(dict[b'labels'])
        # return dict[b'data'].reshape(len(dict[b'data']), 3, 32, 32).transpose(0, 2, 3, 1), np.array(dict[b'labels'])

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)



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

class Detection(data.Dataset):
    """

    """

    def __init__(self, args, train=True, image_sets=['train2017'], transform=None, anno_transform=None,
                 full_test=False):

        self.dataset = args.dataset
        self.root = args.root + '/' + args.data_path + '/'
        self.data_type = args.data_path.split('/')[-1]


        self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.image_loader = LoadImage()
        self.classes, self.ids, self.print_str, self.idlist = self._map(image_sets)


    def _map(self, subsets):

        with open(self.rootpath + 'annots.json', 'r') as f:
            db = json.load(f)

        img_list = []
        names = []
        cls_list = db['classes']
        annots = db['annotations']
        idlist = []
        if 'ids' in db.keys():
            idlist = db['id']
        ni = 0
        nb = 0.0
        print_str = ''
        for img_id in annots.keys():
            # pdb.set_trace()
            if annots[img_id]['set'] in subsets:
                names.append(img_id)
                boxes = []
                labels = []
                for anno in annots[img_id]['annos']:
                    nb += 1
                    boxes.append(anno['bbox'])
                    labels.append(anno['label'])
                # print(labels)
                img_list.append([annots[img_id]['set'], img_id, np.asarray(boxes).astype(np.float32),
                                 np.asarray(labels).astype(np.int64)])
                ni += 1

        print_str = '\n\n*Num of images {:d} num of boxes {:d} avergae {:01f}\n\n'.format(ni, int(nb), nb / ni)

        return cls_list, img_list, print_str, idlist

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]
        subset_str = annot_info[0]
        img_id = annot_info[1]
        boxes = annot_info[2]
        labels = annot_info[3]

        img_name = '{:s}{:s}.jpg'.format(self.root, img_id)
        # print(img_name)

        # t0 = time.perf_counter()
        img = self.image_loader(img_name)
        height, width, _ = img.shape
        wh = [width, height]
        # print('t1', time.perf_counter()-t0)
        # t0 = time.perf_counter()

        imgs = []
        imgs.append(img)
        imgs = np.asarray(imgs)
        # print(img_name, imgs.shape)
        imgs, boxes_, labels_ = self.transform(imgs, boxes, labels, 1)
        # print('t2', time.perf_counter()-t0)
        target = np.hstack((boxes_, np.expand_dims(labels_, axis=1)))
        # imgs = imgs[:, :, :, (2, 1, 0)]
        imgs = imgs[0]
        # images = torch.from_numpy(imgs).permute(2, 0, 1)
        # print(imgs.size(), boxes_, labels_)
        # prior_labels, prior_gt_locations = torch.rand(1,2), torch.rand(2)

        # if self.anno_transform:
        #     prior_labels, prior_gt_locations = self.anno_transform(boxes_, labels_, len(labels_))

        return imgs, target, index, wh


class BaseTransform:
    def __init__(self, size, means, stds):
        self.size = size
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, seq_len=1):
        return base_transform_nimgs(image, self.size, self.means, self.stds, seq_len=seq_len), boxes, labels


def base_transform_nimgs(images, size, mean, stds, seq_len=1):
    res_imgs = []
    # print(images.shape)
    for i in range(seq_len):
        # img = Image.fromarray(images[i,:, :, :])
        # img = img.resize((size, size), Image.BILINEAR)
        img = cv2.resize(images[i, :, :, :], (size, size)).astype(np.float32)
        #img = images[i, :, :, :].astype(np.float32)
        # img  = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        res_imgs += [torch.from_numpy(img).permute(2, 0, 1)]
    # pdb.set_trace()
    # res_imgs = np.asarray(res_imgs)
    return [F.normalize(img_tensor, mean, stds) for img_tensor in res_imgs]