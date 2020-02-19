import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)
import files
from tensorboardX import SummaryWriter
import os
import numpy as np
import time

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as tfs

from data import DataSet,return_model_loader
from util import weight_init, write_conv, setup_runtime, AverageMeter, MovingAverage


def RotationDataLoader(image_dir, is_validation=False,
                       batch_size=256,  crop_size=224, num_workers=4,shuffle=True):

    normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = tfs.Compose([
                                tfs.RandomResizedCrop(crop_size),
                                tfs.RandomGrayscale(p=0.2),
                                tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                tfs.RandomHorizontalFlip(),
                                tfs.Lambda(lambda img: torch.stack([normalize(tfs.ToTensor()(
                                    tfs.functional.rotate(img, angle))) for angle in [0, 90, 180, 270]]
                                ))
                            ])
    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader

class Optimizer:
    def __init__(self):
        self.num_epochs = 30
        self.lr = 0.05
        self.lr_schedule = lambda epoch: (self.lr * (0.1 ** (epoch//args.lrdrop)))*(epoch<80) + (epoch>=80)*self.lr*(0.1**3)
        self.momentum = 0.9
        self.weight_decay = 10**(-5)

        self.resume = True
        self.checkpoint_dir = None
        self.writer = None

        self.K = args.ncl
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.val_loader = RotationDataLoader(args.imagenet_path, is_validation=True,
                                             batch_size=args.batch_size, num_workers=args.workers,shuffle=True)


    def optimize_epoch(self, model, optimizer, loader, epoch, validation=False):
        print(f"Starting epoch {epoch}, validation: {validation} " + "="*30)
        loss_value = AverageMeter()
        rotacc_value = AverageMeter()

        # house keeping
        if not validation:
            model.train()
            lr = self.lr_schedule(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            model.eval()

        XE = torch.nn.CrossEntropyLoss().to(self.dev)
        l_dl = 0  # len(loader)
        now = time.time()
        batch_time = MovingAverage(intertia=0.9)
        for iter, (data, label, selected) in enumerate(loader):
            now = time.time()

            if not validation:
                niter = epoch * len(loader.dataset) + iter*args.batch_size
            data = data.to(self.dev)
            mass = data.size(0)
            where = np.arange(mass,dtype=int) * 4
            data = data.view(mass * 4, 3, data.size(3), data.size(4))
            rotlabel = torch.tensor(range(4)).view(-1, 1).repeat(mass, 1).view(-1).to(self.dev)
            #################### train CNN ###########################################
            if not validation:
                final = model(data)
                if args.onlyrot:
                    loss = torch.Tensor([0]).to(self.dev)
                else:
                    if args.hc == 1:
                        loss = XE(final[0][where], self.L[selected])
                    else:
                        loss = torch.mean(torch.stack([XE(final[k][where], self.L[k, selected]) for k in range(args.hc)]))
                rotloss = XE(final[-1], rotlabel)
                pred = torch.argmax(final[-1], 1)

                total_loss = loss + rotloss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                correct = (pred == rotlabel).to(torch.float)
                rotacc = correct.sum() / float(mass)
            else:
                final = model(data)
                pred = torch.argmax(final[-1], 1)
                correct = (pred == rotlabel.cuda()).to(torch.float)
                rotacc = correct.sum() / float(mass)
                total_loss = torch.Tensor([0])
                loss = torch.Tensor([0])
                rotloss = torch.Tensor([0])
            rotacc_value.update(rotacc.item(), mass)
            loss_value.update(total_loss.item(), mass)

            batch_time.update(time.time() - now)
            now = time.time()
            print(
                f"Loss: {loss_value.avg:03.3f}, RotAcc: {rotacc_value.avg:03.3f} | {epoch: 3}/{iter:05}/{l_dl:05} Freq: {mass / batch_time.avg:04.1f}Hz:",
                end='\r', flush=True)

            # every few iter logging
            if (iter % args.logiter == 0):
                if not validation:
                    print(niter, " Loss: {0:.3f}".format(loss.item()), flush=True)
                    with torch.no_grad():
                        if not args.onlyrot:
                            pred = torch.argmax(final[0][where], dim=1)
                            pseudoloss = XE(final[0][where], pred)
                    if not args.onlyrot:
                        self.writer.add_scalar('Pseudoloss', pseudoloss.item(), niter)
                    self.writer.add_scalar('lr', self.lr_schedule(epoch), niter)
                    self.writer.add_scalar('Loss', loss.item(), niter)
                    self.writer.add_scalar('RotLoss', rotloss.item(), niter)
                    self.writer.add_scalar('RotAcc', rotacc.item(), niter)

                    if iter > 0:
                        self.writer.add_scalar('Freq(Hz)', mass/(time.time() - now), niter)

        # end of epoch logging
        if self.writer and (epoch % self.log_interval == 0):
            write_conv(self.writer, model, epoch)
            if validation:
                print('val Rot-Acc: ', rotacc_value.avg)
                self.writer.add_scalar('val Rot-Acc', rotacc_value.avg, epoch)

        files.save_checkpoint_all(self.checkpoint_dir, model, args.arch,
                                    optimizer,  self.L, epoch,lowest=False)
        return {'loss': loss_value.avg}

    def optimize(self, model, train_loader):
        """Perform full optimization."""
        first_epoch = 0
        model = model.to(self.dev)
        self.optimize_times = [0]
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    weight_decay=self.weight_decay,
                                    momentum=self.momentum,
                                    lr=self.lr)
        if self.checkpoint_dir is not None and self.resume:
            self.L, first_epoch = files.load_checkpoint_all(self.checkpoint_dir, model=None, opt=None)
            print('loaded from: ', self.checkpoint_dir,flush=True)
            print('first five entries of L: ', self.L[:5], flush=True)
            print('found first epoch to be', first_epoch, flush=True)
            first_epoch = 0
            self.optimize_times = [0]
            self.L = self.L.cuda()
            print("model.headcount ", model.headcount, flush=True)


        #####################################################################################
        # Perform optmization ###############################################################
        lowest_loss = 1e9
        epoch = first_epoch
        while epoch < (self.num_epochs+1):
            if not args.val_only:
                m = self.optimize_epoch(model, optimizer, train_loader, epoch, validation=False)
                if m['loss'] < lowest_loss:
                    lowest_loss = m['loss']
                    files.save_checkpoint_all(self.checkpoint_dir, model, args.arch,
                                              optimizer, self.L, epoch, lowest=True)
            else:
                print('='*30 +' doing only validation ' + "="*30)
                epoch = self.num_epochs
            m = self.optimize_epoch(model, optimizer, self.val_loader, epoch, validation=True)
            epoch += 1
        print(f"Model optimization completed. Saving final model to {os.path.join(self.checkpoint_dir, 'model_final.pth.tar')}")
        torch.save(model, os.path.join(self.checkpoint_dir, 'model_final.pth.tar'))
        return model




def get_parser():
    parser = argparse.ArgumentParser(description='Retrain with given labels combined with RotNet loss')
    # optimizer
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of epochs')
    parser.add_argument('--batch-size', default=64, type=int, metavar='BS', help='batch size')
    parser.add_argument('--lr', default=0.05, type=float, metavar='FLOAT', help='initial learning rate')
    parser.add_argument('--lrdrop', default=30, type=int, metavar='INT', help='multiply LR by 0.1 every')

    # architecture
    parser.add_argument('--arch', default='alexnet', type=str, help='alexnet or resnet')
    parser.add_argument('--archspec', default='big', type=str, help='big or small for alexnet ')
    parser.add_argument('--ncl', default=1000, type=int, metavar='INT', help='number of clusters')
    parser.add_argument('--hc', default=1, type=int, metavar='INT', help='number of heads')
    parser.add_argument('--init', default=False, action='store_true', help='initialization of network to PyTorch 0.4')

    # what we do in this code
    parser.add_argument('--val-only', default=False, action='store_true', help='if we run only validation set')
    parser.add_argument('--onlyrot', default=False, action='store_true', help='if train only RotNet')

    # housekeeping
    parser.add_argument('--data', default="Imagenet", type=str)
    parser.add_argument('--device', default="0", type=str, metavar='N', help='GPU device')
    parser.add_argument('--exp', default='./rot-retrain', metavar='DIR', help='path to result dirs')
    parser.add_argument('--workers', default=6, type=int, metavar='N', help='number workers (default: 6)')
    parser.add_argument('--imagenet-path', default='/home/ubuntu/data/imagenet', type=str, help='')
    parser.add_argument('--comment', default='rot-retrain', type=str, help='comment for tensorboardX')
    parser.add_argument('--log-interval', default=1, type=int, metavar='INT', help='save stuff every x epochs')
    parser.add_argument('--logiter', default=200, type=int, metavar='INT', help='log every x-th batch')

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    name = "%s" % args.comment.replace('/', '_')
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    setup_runtime(seed=42, cuda_dev_id=args.device)
    print(args, flush=True)
    print()
    print(name,flush=True)

    writer = SummaryWriter('./runs/%s/%s'%(args.data,name))
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))


    # Setup model and train_loader
    print('Commencing!', flush=True)
    model, train_loader = return_model_loader(args)

    train_loader = RotationDataLoader(args.imagenet_path, is_validation=False,
                                      crop_size=224, batch_size=args.batch_size, num_workers=args.workers,
                                      shuffle=True)

    # add additional head to the network for RotNet loss.
    if args.arch == 'alexnet':
        if args.hc == 1:
            model.__setattr__("top_layer0", nn.Linear(4096, args.ncl))
            model.top_layer = None
        model.headcount = args.hc+1
        model.__setattr__("top_layer%s" % args.hc, nn.Linear(4096, 4))
    else:
        if args.hc == 1:
            model.__setattr__("top_layer0", nn.Linear(2048*int(args.archspec), args.ncl))
            model.top_layer = None
        model.headcount = args.hc+1
        model.__setattr__("top_layer%s" % args.hc, nn.Linear(2048*int(args.archspec), 4))
    if args.init:
        for mod in model.modules():
            mod.apply(weight_init)

    # Setup optimizer
    o = Optimizer()
    o.writer = writer
    o.lr = args.lr
    o.num_epochs = args.epochs
    o.resume = True
    o.log_interval = args.log_interval
    o.checkpoint_dir = os.path.join(args.exp, 'checkpoints')


    # Optimize
    o.optimize(model, train_loader)
