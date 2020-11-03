# Self-labelling via simultaneous clustering and representation learning

🆗🆗🎉 _NEW models (20th August 2020): Added standard SeLa pretrained torchvision ResNet models to make loading much easier + added baselines using better MoCov2 augmentation (~69% LP performance) + added evaluation with K=1000 for ImageNet "unuspervised clustering"_

🆕✅🎉 _updated code: 23rd April 2020: bug fixes + CIFAR code + evaluation for resnet & alexnet._

Checkout our [blogpost](http://www.robots.ox.ac.uk/~vgg/blog/self-labelling-via-simultaneous-clustering-and-representation-learning.html) for a quick non-technical overview and an interactive visualization of our clusters.

## Self-Label

This code is the official implementation of the ICLR 2020 paper [Self-labelling via simultaneous clustering and representation learning](https://openreview.net/forum?id=Hyx-jyBFPr).

### Abstract
Combining clustering and representation learning is one of the most promising
approaches for unsupervised learning of deep neural networks. However, doing
so naively leads to ill posed learning problems with degenerate solutions. In this
paper, we propose a novel and principled learning formulation that addresses
these issues. The method is obtained by maximizing the information between
labels and input data indices. We show that this criterion extends standard crossentropy minimization to an optimal transport problem, which we solve efficiently
for millions of input images and thousands of labels using a fast variant of the
Sinkhorn-Knopp algorithm. The resulting method is able to self-label visual data
so as to train highly competitive image representations without manual labels. Our
method achieves state of the art representation learning performance for AlexNet
and ResNet-50 on SVHN, CIFAR-10, CIFAR-100 and ImageNet.

### Results at a glance

|                     | NMI(%) | aNMI(%) | ARI(%) | LP Acc (%) | 
|---------------------|--------|---------|--------|------------|
| AlexNet 1k          | 50.5   | 12.2    | 2.7    | 42.1       |  
| AlexNet 10k         | 66.4   | 4.7     | 4.7    | 43.8       |  
| R50 10x3k           | 54.2   | 34.4    | 7.2    | 61.5       |  

#### With better augmentations (all single crop)

|                      | Label-Acc  | NMI(%) | aNMI(%) | ARI(%) | LP Acc (%) | model_weights |
|----------------------|------|--------|---------|--------|------------|---------------|
| Aug++ R18  1k (new)  | 26.9 | 62.7   | 36.4    | 12.5   | 53.3       | [here](http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/new_models/resnet18-1k_pp.pth) |
| Aug++ R50  1k (new)  | 30.5 | 65.7   | 42.0    | 16.2   | 63.5       | [here](http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/new_models/resnet50-1k_pp.pth) |
| Aug++ R50 10x3k (new)| 38.1 |75.7   | 52.8    | 27.6   | 68.8       | [here](http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/new_models/resnet50-10x3k_pp.pth) |
|(MoCo-v2 + k-means**, K=3k)      |  |71.4   | 39.6    | 15.8   | 71.1       |               |

* "Aug++" refers to the better augmentations used in SimCLR, taken from the [MoCo-v2 repo](https://github.com/facebookresearch/moco/blob/master/main_moco.py#L225), but I still only trained for 280 epochs, with three lr-drops as in [CMC](https://github.com/HobbitLong/CMC/blob/master/train_CMC.py#L50).
* There are still further improvements to be made with a MLP or training 800 epochs (I train 280), as done in SimCLR, [MoCov2](https://github.com/facebookresearch/moco) and [SwAV](https://github.com/facebookresearch/swav).
* **MoCo-v2 uses 800 epochs, MLP and cos-lr-schedule. On MoCo-v2 I run k-means (K=3000) on the avg-pooled features (after the MLP-head it's pretty much the same performance) to obtain NMI, aNMI and ARI numbers.
* Models above use standard torchvision ResNet backbones so loading is now super easy:
```
import torch, torchvision
model = torchvision.models.resnet50(pretrained=False, num_classes=3000)
ckpt = torch.load('resnet50-10x3k_pp.pth')
model.load_state_dict(ckpt['state_dict'])
pseudolabels = ckpt['L']
```
* note on improvement potential: by just using "aug+": I get LP-accuracy of 67.2% after 200 epochs. MoCo-v2 with "aug+" only has 63.4% after 200 epochs.

## Clusters that were discovered by our method
*Sorted*

![Imagenet validation images with clusters sorted by imagenet purity](https://www.robots.ox.ac.uk/~vgg/research/self-label/asset/sorted-clusters.png)

*Random*

![Imagenet validation images with random clusters](https://www.robots.ox.ac.uk/~vgg/research/self-label/asset/random-clusters.png)

The edge-colors encode the true imagenet classes (which are not used for training).
You can view all clusters [here](http://www.robots.ox.ac.uk/~vgg/blog/self-labelling-via-simultaneous-clustering-and-representation-learning.html).

## Requirements
* Python >3.6
* PyTorch > 1.0
* CUDA
* Numpy, SciPy
* also, see requirements.txt
* (optional:) TensorboardX

## Running our code
Run the self-supervised training of an AlexNet with the command
```
$./scripts/alexnet.sh
```
or train a ResNet-50 with
```
$./scripts/resnet.sh
```
Note: you need to specify your dataset directory (it expects a format just like ImageNet with "train" and "val" folders). You also need to give the code enough GPUs to allow for storage of activations on the GPU. Otherwise you need to use the CPU variant which is significantly slower.

Full documentation of the unsupervised training code `main.py`:
```
usage: main.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
               [--lrdrop LRDROP] [--wd WD] [--dtype {f64,f32}] [--nopts NOPTS]
               [--augs AUGS] [--paugs PAUGS] [--lamb LAMB] [--cpu]
               [--arch ARCH] [--archspec {big,small}] [--ncl NCL] [--hc HC]
               [--device DEVICE] [--modeldevice MODELDEVICE] [--exp EXP]
               [--workers WORKERS] [--imagenet-path IMAGENET_PATH]
               [--comment COMMENT] [--log-intv LOG_INTV] [--log-iter LOG_ITER]

PyTorch Implementation of Self-Label

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs
  --batch-size BATCH_SIZE
                        batch size (default: 256)
  --lr LR               initial learning rate (default: 0.05)
  --lrdrop LRDROP       multiply LR by 0.1 every (default: 150 epochs)
  --wd WD               weight decay pow (default: (-5)
  --dtype {f64,f32}     SK-algo dtype (default: f64)
  --nopts NOPTS         number of pseudo-opts (default: 100)
  --augs AUGS           augmentation level (default: 3)
  --paugs PAUGS         for pseudoopt: augmentation level (default: 3)
  --lamb LAMB           for pseudoopt: lambda (default:25)
  --cpu                 use CPU variant (slow) (default: off)
  --arch ARCH           alexnet or resnet (default: alexnet)
  --archspec {big,small}
                        alexnet variant (default:big)
  --ncl NCL             number of clusters per head (default: 3000)
  --hc HC               number of heads (default: 1)
  --device DEVICE       GPU devices to use for storage and model
  --modeldevice MODELDEVICE
                        GPU numbers on which the CNN runs
  --exp EXP             path to experiment directory
  --workers WORKERS     number workers (default: 6)
  --imagenet-path IMAGENET_PATH
                        path to folder that contains `train` and `val`
  --comment COMMENT     name for tensorboardX
  --log-intv LOG_INTV   save stuff every x epochs (default: 1)
  --log-iter LOG_ITER   log every x-th batch (default: 200)
```

## Evaluation
### Linear Evaluation
We provide the linear evaluation methods in this repo.
Simply download the models via `. ./scripts/download_models.sh` and then either run `scripts/eval-alexnet.sh` or `scripts/eval-resnet.sh`.

### Pascal VOC
We follow the standard evaluation protocols for self-supervised visual representation learning.
* for Classification: we follow the PyTorch implementation of [DeepCluster](https://github.com/facebookresearch/deepcluster) with frozen BatchNorm.
* for Segmentation: we follow the implmentation from the [Colorization paper](https://github.com/richzhang/colorization) which uses the [FCN repo](https://github.com/shelhamer/fcn.berkeleyvision.org). Note: requires the Caffe framework
* for Detection: we follow [Krähenbühl et al.'s implementation](https://www.philkr.net/2001/02/01/pub/)
based on the [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn). Note: requires the Caffe framework

## Our extracted pseudolabels
As we show in the paper, the pseudolabels we generate from our training can be used to quickly train a neural network with regular cross-entropy.
Moreover they seem to correctly group together similar images. Hence we provide the labels for everyone to use.
### AlexNet
You can download the pseudolabels from our best (raw) AlexNet model with 10x3000 clusters [here](http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/alexnet-labels.csv).
### ResNet
You can download the pseudolabels from our best ResNet model with 10x3000 clusters [here](http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/resnet-labels.csv).

## Trained models
You can also download our trained models by running
```
$./scripts/download_models.sh
```
Use them like this:
```
import torch
import models
d = torch.load('self-label_models/resnet-10x3k.pth')
m = models.resnet(num_classes = [3000]*10)
m.load_state_dict(d)

d = torch.load('self-label_models/alexnet-10x3k-wRot.pth')
m = models.alexnet(num_classes = [3000]*10)
m.load_state_dict(d)

```

## Reference

If you use this code etc., please cite the following paper:

Yuki M. Asano, Christian Rupprecht and Andrea Vedaldi.  "Self-labelling via simultaneous clustering and representation learning." Proc. ICLR (2020)

```
@inproceedings{asano2020self,
  title={Self-labelling via simultaneous clustering and representation learning},
  author={Asano, Yuki M. and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
}
```
