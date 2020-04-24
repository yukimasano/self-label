import torchvision
import torch
import torchvision.transforms as tfs
import models
import os
import util

class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,
                       batch_size=256, image_size=256, crop_size=224,
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       num_workers=8,
                       augs=1, shuffle=True):

    print(image_dir)
    if image_dir is None:
        return None

    print("imagesize: ", image_size, "cropsize: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)
    if augs == 0:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 1:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 2:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 3:
        _transforms = tfs.Compose([
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomGrayscale(p=0.2),
                                    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])

    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader


def return_model_loader(args, return_loader=True):
    outs = [args.ncl]*args.hc
    assert args.arch in ['alexnet','resnetv2','resnetv1']
    if args.arch == 'alexnet':
        model = models.__dict__[args.arch](num_classes=outs)
    elif args.arch == 'resnetv2':  # resnet
        model = models.__dict__[args.arch](num_classes=outs, nlayers=50, expansion=1)
    else:
        model = models.__dict__[args.arch](num_classes=outs)
    if not return_loader:
        return model
    train_loader = get_aug_dataloader(image_dir=args.imagenet_path,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers,
                                      augs=int(args.augs))

    return model, train_loader

def get_standard_data_loader(image_dir, is_validation=False,
                             batch_size=192, image_size=256, crop_size=224,
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                             num_workers=8,no_random_crops=False, tencrops=True):
    """Get a standard data loader for evaluating AlexNet representations in a standard way.
    """
    if image_dir is None:
        return None
    normalize = tfs.Normalize(mean=mean, std=std)
    if is_validation:
        if tencrops:
             transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.TenCrop(crop_size),
                tfs.Lambda(lambda crops: torch.stack([normalize(tfs.ToTensor()(crop)) for crop in crops]))
            ])
             batch_size = int(batch_size/10)
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.ToTensor(),
                normalize
            ])
    else:
        if not no_random_crops:
            transforms = tfs.Compose([
                tfs.RandomResizedCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])

    dataset = torchvision.datasets.ImageFolder(image_dir, transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None
    )
    return loader

def get_standard_data_loader_pairs(dir_path, **kargs):
    """Get a pair of data loaders for training and validation.
         This is only used for the representation EVALUATION part.
    """
    train = get_standard_data_loader(os.path.join(dir_path, "train"), is_validation=False, **kargs)
    val = get_standard_data_loader(os.path.join(dir_path, "val"), is_validation=True, **kargs)
    return train, val