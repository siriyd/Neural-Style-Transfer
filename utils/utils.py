# write the dataset class

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(self.root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_transform(size,final_size,crop):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.CenterCrop(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_feat,style_feat):
    #[batch size, channels, height, width]
    size = content_feat.size()
    style_mean,style_std = calculate_mean_std(style_feat)
    content_mean,content_std = calculate_mean_std(content_feat)
    normaked_content_feat = (content_feat-content_mean.expand(size))/(content_std.expand(size)+1e-5)
    return normaked_content_feat*style_std.expand(size)+style_mean.expand(size)

def calculate_mean_std(feat,eps=1e-5):
    #[batch size, channels, height, width]
    size = feat.size()
    assert (len(size) == 4)
    batch_size , channels = size[:2]
    feat_mean = feat.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
    feat_var=feat.view(batch_size,channels,-1).var(dim=2,unbiased=False)+eps
    feat_std = feat_var.sqrt().view(batch_size,channels,1,1)
    return feat_mean,feat_std
