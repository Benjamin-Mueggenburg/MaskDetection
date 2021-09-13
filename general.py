from torch.nn.functional as F
import torch 
from torchvision.transforms import transforms

def rescale_img(image, size, mode="nearest"):
    return transforms.Resize(size).forward(image)

device = torch.device('cuda')