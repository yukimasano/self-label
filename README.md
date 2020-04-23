# Self-labelling via simultaneous clustering and representation learning

🆕✅🎉 _updated code: 23rd April: bug fixes + CIFAR code + evaluation for resnet + Alexnet._

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

## clusters that were discovered by our method
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
  author={Asano YM. and Rupprecht C. and Vedaldi A.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
}
```
