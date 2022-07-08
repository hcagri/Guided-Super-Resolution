import torch
from torchvision.transforms import transforms as T
import numpy as np 
from PIL import Image

def create_grid(N, normalize = True):
    x = torch.arange(0, N, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
    pos_image = torch.stack([grid_x, grid_y], dim=0)
    
    if normalize:
        pos_image = (pos_image-x.mean())/x.max()

    return pos_image

def unnormalize(mean, std, img):
    # First multiply with std then add mean
    unnorm_trans = T.Compose([
        T.Normalize(mean=(0), std=(1/std)),
        T.Normalize(mean=-mean, std=(1))
    ])
    return unnorm_trans(img)

def normalize(img):

    mean = img.mean(dim=[1,2])
    std = img.std(dim=[1,2])

    normalized_img = T.Normalize(mean=mean, std=std)(img)
    return mean, std, normalized_img


def save_image(tensor, save_path):
    arr = tensor.squeeze().detach().cpu().numpy()
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(save_path)


def percent_bad_pixels(output, target, sigma = 1):
    ''' Output and Target must be de-normalized, [0-255]'''
    diff = abs(output - target)
    PBP = torch.where(diff > sigma, 1, 0).float().mean()
    return PBP

