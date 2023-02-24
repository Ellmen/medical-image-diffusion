import torch
import torchvision 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from utils.get_data import get_data
from utils.metrics import get_dice_arr
from utils.segmentation import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow(img, path):
    # TODO: unnormalize if needed
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path)


def run_tests(models, names, testdataloader):
    dataiter = iter(testdataloader)
    images, masks = next(dataiter)
    imshow(torchvision.utils.make_grid(images), 'outputs/test_retinas.png')
    data, label = images.to(device), masks.to(device)
    for model, name in zip(models, names):
        model.eval()
        out = model(data)
        softmax_out = F.sigmoid(out)
        with open(f'outputs/{name}_out_sample.txt', 'w') as f:
            f.write(str(out[0][0][50]))
        with open(f'outputs/{name}_out_dice.txt', 'w') as f:
            f.write(str(get_dice_arr(out, masks)))
        imshow(torchvision.utils.make_grid(softmax_out.detach().cpu()), f'outputs/{name}.png')


def test_models(data_path, m1_path, m2_path, m3_path, out_path):
    traindataloader, valdataloader, testdataloader = get_data(data_path)
    model1 = UNet(channel_in=1, channel_out=1)
    model2 = UNet(channel_in=1, channel_out=1)
    model3 = UNet(channel_in=1, channel_out=1)
    model1.load_state_dict(torch.load(m1_path))
    model2.load_state_dict(torch.load(m2_path))
    model3.load_state_dict(torch.load(m3_path))
    run_tests(
        [
            model1,
            model2,
            model3,
        ],
        [
            '3_epochs_0.1_momentum',
            '15_epochs_0.1_momentum',
            '3_epochs_0.25_momentum',
        ],
        testdataloader
    )


test_models(
    snakemake.input[0],
    snakemake.input[1],
    snakemake.input[2],
    snakemake.input[3],
    snakemake.output[0]
)
