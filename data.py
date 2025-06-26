import os
import os.path
import cv2
import glob
import scipy.io
import numpy as np
import torch
from torchvision import transforms

DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn

@register_dataset("ODT")
def load_ODT(data, target, batch_size=4, num_workers=4, img_size=64, sp=2):
    train_dataset = ODT_train(data, target, img_size=img_size, sp=sp)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    return train_loader

class ODT_train(torch.utils.data.Dataset):
    def __init__(self, data_path, target_path, img_size=64, sp=2):
        super().__init__()
        self.data_path = os.path.join(data_path + '_split', 'train')
        self.target_path = os.path.join(target_path + '_split', 'train')
        self.img_size = img_size
        self.len = 0
        self.bounds = [0]
        self.nWs = []
        self.sp=sp
        self.max_sp = 256 // img_size

        if self.sp > self.max_sp:
            raise('exceed maximum downsampling factor')
        
        self.num_patch = self.max_sp // self.sp

        self.folders = sorted([x for x in glob.glob(os.path.join(self.data_path, '*')) if os.path.isdir(x)])
        self.target_folders = sorted([x for x in glob.glob(os.path.join(self.target_path, '*')) if os.path.isdir(x)])

        self.volume_ids = []

        for i, folder in enumerate(self.folders):
            self.volume_ids.append(i)
            files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            self.len += len(files)
            self.bounds.append(self.len)
        
        self.len = self.len * self.num_patch

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        part = index % self.num_patch
        index = index // self.num_patch
    
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                target_folder = self.target_folders[i-1]
                index -= self.bounds[i-1]
                break

        assert folder.split(os.path.sep)[-1].split('_')[0] == target_folder.split(os.path.sep)[-1].split('_')[0]
        assert len(os.listdir(folder)) == len(os.listdir(target_folder))

        data_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        target_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))
        assert os.path.basename(data_files[index])[:-4] == os.path.basename(target_files[index])[:-4]

        input_path = data_files[index]
        input_data = self.load_input_data(input_path, part)

        target_path = target_files[index]
        target_img = self.load_target_img(target_path, part)

        return input_data, target_img
    
    def load_input_data(self, input_path, part=None):
        rawdata = np.load(input_path).astype('float32')

        if self.num_patch > 1:
            rawdata = rawdata[:, :, part*self.img_size*self.sp:(part+1)*self.img_size*self.sp]

        mask = torch.arange(0, rawdata.shape[-1], self.sp)
        input_data = rawdata[:, :, mask]

        assert input_data.shape[-1] == rawdata.shape[-1] // self.sp
        assert input_data.shape[-1] == self.img_size
        assert input_data.shape[0] == 4
        return input_data

    def load_target_img(self, target_path, part=None):
        target_img = cv2.imread(target_path, -1)

        if self.num_patch > 1:
            target_img = target_img[:, part*self.img_size*self.sp:(part+1)*self.img_size*self.sp]

        target_img = target_img[None, :, :].astype('float')
        target_img = torch.from_numpy(target_img / 65535).float()
        return target_img


@register_dataset("ODT_val")
def load_ODT_val(data, target, num_workers=1, sp=2):
    valid_dataset = ODT_val(data, target, sp=sp)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return valid_loader

class ODT_val(torch.utils.data.Dataset):
    def __init__(self, data_path, target_path, sp=2):
        super().__init__()
        self.data_path = os.path.join(data_path, 'test')
        self.target_path = os.path.join(target_path, 'test')
        self.len = 0
        self.bounds = [0]
        self.sp = sp

        self.folders = sorted([x for x in glob.glob(os.path.join(self.data_path, '*')) if os.path.isdir(x)])
        self.target_folders = sorted([x for x in glob.glob(os.path.join(self.target_path, '*')) if os.path.isdir(x)])

        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            self.len += len(files)
            self.bounds.append(self.len)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        for i, bound in enumerate(self.bounds):
            if index < bound:
                folder = self.folders[i-1]
                target_folder = self.target_folders[i-1]
                index -= self.bounds[i-1]
                break

        assert folder.split(os.path.sep)[-1].split('_')[0] == target_folder.split(os.path.sep)[-1].split('_')[0]
        assert len(os.listdir(folder)) == len(os.listdir(target_folder))
        
        data_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        target_files = sorted(glob.glob(os.path.join(target_folder, "*.tif")))
        assert os.path.basename(data_files[index])[:-4] == os.path.basename(target_files[index])[:-4]

        rawdata = np.load(data_files[index]).astype('float32')
        raw_width = rawdata.shape[-1]
        mask = torch.arange(0, rawdata.shape[-1], self.sp)
        input_data = rawdata[:, :, mask]

        assert input_data.shape[-1] == rawdata.shape[-1] // self.sp
        assert input_data.shape[0] == 4

        target_img = cv2.imread(target_files[index], -1)
        target_img = target_img[None, :, :].astype('float')
        target_img = torch.from_numpy(target_img / 65535).float()

        _, fname = os.path.split(data_files[index])
        data_name = fname

        return input_data, data_name, target_img, raw_width