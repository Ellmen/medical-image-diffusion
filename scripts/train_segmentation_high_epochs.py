import torch
import torchvision 
from torch import nn
import numpy as np
print(torch.__version__)

import random
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

from utils.get_data import get_data
from utils.segmentation import UNet
from utils.training import train_model


def train_segmentation_alt(data_path, out_path):

    traindataloader, valdataloader, testdataloader = get_data(data_path)

    print('Number of training batches:', len(traindataloader))
    print('Number of validation batches:', len(valdataloader))
    print('Number of testing batches:', len(testdataloader))

    model = UNet(channel_in=1, channel_out=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epochs=15
    train_model(
        model,
        device,
        epochs,
        out_path,
        traindataloader,
        valdataloader
    )

train_segmentation_alt(snakemake.input[0], snakemake.output[0])
