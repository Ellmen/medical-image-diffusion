import numpy as np

from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .segmentation import SegmentationData


def get_data(drive_dir):
    # Step: Split between train, val, and test from the overall trainset
    transform = transforms.Compose([transforms.ToTensor()])
    # transforms.Resize((256,256))
    trainset = SegmentationData(root=f'{drive_dir}/training/', transforms=transform)
    testset = SegmentationData(root=f'{drive_dir}/test/', transforms=transform)

    val_percentage = 0.2
    num_train = len(trainset)
    num_test = len(testset)

    train_indices = list(range(num_train))
    test_indices = list(range(num_test))
    split = int(np.floor(val_percentage * num_train))

    train_idx, valid_idx = train_indices[split:], train_indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_indices)

    batch_size = 4 # create batch-based on how much memory you have and your data size

    traindataloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    valdataloader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    testdataloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=0)
    return traindataloader, valdataloader, testdataloader
