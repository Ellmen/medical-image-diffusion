import torch
import torchvision 
from torch import nn
import torch.nn.functional as F
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

traindataloader, valdataloader, testdataloader = get_data('DRIVE')

print('Number of training samples:', len(traindataloader))
print('Number of validation samples:', len(valdataloader))
print('Number of testing samples:', len(testdataloader))


# function to unnormalise images and using transpose to change order to [H, W, Channel] 
def imshow(img):
    return
    # TODO: unnormalize if needed
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


import matplotlib.pyplot as plt 

# always check the shape of your training data
dataiter = iter(traindataloader)
images, masks = next(dataiter)

# show images 
imshow(torchvision.utils.make_grid(images))
imshow(torchvision.utils.make_grid(masks))


model = UNet(channel_in=1, channel_out=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from utils.metrics import get_dice_arr
from utils.training import train_model

# train_model()

# In[ ]:

# Apply on your 10% validaton data and report your result
ckptFileName = 'UNet_CKPT_best.pt'
# load the saved weights
model.load_state_dict(torch.load(ckptFileName))


### TODO: compute the accuracy of your model on validation set --> You are required to use dice similarity coefficient

total = 0
model.eval()

dataiter = iter(valdataloader)
# dataiter = iter(traindataloader)
images, masks = next(dataiter)
data, label = images.to(device), masks.to(device)
out = model(data)
print(out[0][0][50])
# output = model(images.to(device))
output = model(images).to(device)
print(output[0][0][50])
print(get_dice_arr(output, masks))
# show images 
imshow(torchvision.utils.make_grid(images))
imshow(torchvision.utils.make_grid(output.detach().cpu()))

