import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

#############################################
### Dataset class for RGB imagery #####
#############################################

class LivenessDataset(Dataset):
    def __init__(self, num_context_frames, num_target_frames, 
                 data_path, augmentations=None):
        # L: Length of data, F: Number of frames
        arr = np.load(data_path).astype(np.uint8)  # (58, 10, 852, 480, 3) L x F x H x W x C

        # self.arr = arr
        self.context_frames = arr[:, :num_context_frames, :, :, :]
        self.target_frames = arr[:, num_context_frames:num_context_frames+num_target_frames, :, :, :]

        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames

        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.context_frames.shape[0]
    
    def __getitem__(self, idx):
        # Shape that is required for Conv3D: N x C x F x H x W
        # Desired output shape: C x F x H x W
        # self.arr shape: L x F x H x W x C
        # ToTensor()'s conversion: H x W x C -> C x H x W
        
        f, h, w, c = self.context_frames[0].shape
        context_frames = torch.zeros(3, self.num_context_frames, h, w)
        target_frames = torch.zeros(3, self.num_target_frames, h, w)

        for i in range(self.num_context_frames):
            frame = self.context_frames[idx, i, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            context_frames[:, i, :, :] = ts

        for j in range(self.num_target_frames):
            frame = self.target_frames[idx, j, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            target_frames[:, j, :, :] = ts
        
        return context_frames, target_frames
    
class LivenessDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, 
                 num_context_frames, num_target_frames,
                 data_path,
                 split_ratio=[0.7, 0.15, 0.15]):
        super().__init__()
        self.batch_size = batch_size
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.data_path = data_path
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = LivenessDataset(self.num_context_frames,
                                       self.num_target_frames,
                                       self.data_path)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        split[2] = len(full_dataset) - sum(split[:2])
        
        self.split = split
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)    

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)    

#############################################
### Dataset class for grayscale imagery #####
#############################################

class LivenessDatasetGrayscale(Dataset):
    def __init__(self, num_context_frames, num_target_frames, data_path):
        # L: Length of data, F: Number of frames
        arr = np.load(data_path).astype(np.uint8)  # (58, 10, 852, 480, 3) L x F x H x W x C

        self.arr = arr
        self.context_frames = arr[:, :num_context_frames, :, :, :]
        self.target_frames = arr[:, num_context_frames:num_context_frames+num_target_frames, :, :, :]

        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1)
        ])
        # self.transform = transforms.ToTensor()

    def __len__(self):
        return self.context_frames.shape[0]
    
    def __getitem__(self, idx):
        # Shape that is required for Conv3D: N x C x F x H x W
        # Desired output shape: C x F x H x W
        # self.arr shape: L x F x H x W x C
        # ToTensor()'s conversion: H x W x C -> C x H x W
        
        f, h, w, c = self.context_frames[0].shape
        context_frames = torch.zeros(1, self.num_context_frames, h, w)
        target_frames = torch.zeros(1, self.num_target_frames, h, w)

        for i in range(self.num_context_frames):
            frame = self.context_frames[idx, i, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            context_frames[:, i, :, :] = ts

        for j in range(self.num_target_frames):
            frame = self.target_frames[idx, j, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            target_frames[:, j, :, :] = ts
        
        # C, F, H, W
        return context_frames, target_frames
    
class LivenessDataModuleGrayscale(pl.LightningDataModule):
    def __init__(self, batch_size, num_context_frames, num_target_frames, data_path,
                 split_ratio=[0.7, 0.15, 0.15]):
        super().__init__()
        self.batch_size = batch_size
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.data_path = data_path
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = LivenessDatasetGrayscale(self.num_context_frames,
                                       self.num_target_frames,
                                       self.data_path)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        split[2] = len(full_dataset) - sum(split[:2])
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)    

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)        
        