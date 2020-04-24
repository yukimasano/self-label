import os
import time
import argparse
import warnings

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import util
import files
import data
from util import TotalAverage, MovingAverage, accuracy

warnings.simplefilter("ignore", UserWarning)

class StandardOptimizer():
    def __init__(self, weight_decay=1e-4):
        self.num_epochs = 40
        self.lr = 0.1
        self.lr_schedule = lambda epoch : self.lr * (0.95 ** (epoch))
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.validate_only = False
        self.resume = True
        self.checkpoint_dir = None
        self.writer = None
        self.dev = torch.device('cuda:0')

    def optimize(self, model, train_loader, val_loader=None, optimizer=None):
        """Perform full optimization."""
        # Initialize
        criterion = self.criterion
        metrics = {'train':[], 'val':[]}
        first_epoch = 0

        # Send models to device
        criterion = criterion.to(self.dev)
        model = model.to(self.dev)

        # Get optimizer (after sending to device)
        if optimizer is None:
            optimizer = self.get_optimizer(model)
        if self.checkpoint_dir is not None:
            model_path = os.path.join(self.checkpoint_dir, 'model.pth')
            if self.resume:
                first_epoch, metrics = files.load_checkpoint(self.checkpoint_dir, model, optimizer)

        # Perform epochs
        if not self.validate_only:
            for epoch in range(first_epoch, 1 if self.validate_only else self.num_epochs):
                print(optimizer)
                m = self.optimize_epoch(model, criterion, optimizer, train_loader, epoch, is_validation=False)
                metrics["train"].append(m)
                if (epoch > (self.num_epochs - 20)) or (epoch % 5 == 0):
                    if val_loader:
                        with torch.no_grad():
                            m = self.optimize_epoch(model, criterion, optimizer, val_loader, epoch, is_validation=True)
                            metrics["val"].append(m)
                files.save_checkpoint(self.checkpoint_dir, model, optimizer, metrics, epoch)
                if epoch in [84, 126]:
                    files.save_checkpoint(self.checkpoint_dir, model, optimizer, metrics, epoch,defsave=True)
        else:
            print('only evaluating!', flush=True)
            with torch.no_grad():
                m = self.optimize_epoch(model, criterion, optimizer, val_loader, 99, is_validation=True)
                metrics["val"].append(m)

        torch.save(model, os.path.join(self.checkpoint_dir, 'model.pth'))

        return model, metrics


    def get_optimizer(self, model):
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.top_layer.parameters()),
                               lr=self.lr_schedule(0),
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)


    def optimize_epoch(self, model, criterion, optimizer, loader, epoch, is_validation=False):
        top1 = []
        top5 = []
        loss_value = []
        top1.append(TotalAverage())
        top5.append(TotalAverage())
        loss_value.append(TotalAverage())
        batch_time = MovingAverage(intertia=0.9)
        now = time.time()

        if is_validation is False:
            model.train()
            lr = self.lr_schedule(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            print("Starting epoch %s" % epoch)
        else:
            model.eval()
        l_dl = len(loader)
        for iter, q in enumerate(loader):
            if len(q) == 3:
                input, label, _s = q
            else:
                input, label = q
            input = input.to(self.dev)
            label = label.to(self.dev)
            mass = input.size(0)
            if is_validation and args.tencrops:
                bs, ncrops, c, h, w = input.size()
                input_tensor = input.view(-1, c, h, w)
                input = input_tensor.to(self.dev)
                predictions = model(input)
                predictions = torch.squeeze(predictions.view(bs, ncrops, -1).mean(1))
            else:
                input = input.to(self.dev)
                predictions = model(input)

            loss = criterion(predictions, label)
            top1_, top5_ = accuracy(predictions, label, topk=(1, 5))
            top1[0].update(top1_.item(), mass)
            top5[0].update(top5_.item(), mass)
            loss_value[0].update(loss.item(), mass)

            if is_validation is False:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_time.update(time.time() - now)
            now = time.time()
            if iter % 50 == 0 :
                print(f"{'V' if is_validation else 'T'} Loss: {loss_value[0].avg:03.3f} "
                      f"Top1: {top1[0].avg:03.1f} Top5: {top5[0].avg:03.1f} "
                      f"{epoch: 3}/{iter:05}/{l_dl:05} Freq: {mass/batch_time.avg:04.1f}Hz:",
                      end='\r', flush=True
                      )
        if is_validation:
            print("validation")
            print("val-top1: %s" % top1[0].avg)
            print("val-top5: %s" % top5[0].avg)
        if self.writer:
            str_ = 'LP/val' if is_validation else 'LP/train'
            self.writer.add_scalar(f'{str_}/top1', top1[0].avg, epoch)
            self.writer.add_scalar(f'{str_}/top5', top5[0].avg, epoch)
            self.writer.add_scalar(f'{str_}/Freq', mass/batch_time.avg, epoch)

        return {"loss": [x.avg for x in loss_value],
                "top1": [x.avg for x in top1],
                "top5": [x.avg for x in top1]}



def get_parser():
    parser = argparse.ArgumentParser(description='Driver')
    parser.add_argument('--device', nargs='+', default="3", type=str, metavar='N', help='use "0 1" for specifying')

    # model
    parser.add_argument('--arch', default='resnetv2', metavar='NAME', help='architecture to train')
    parser.add_argument('--ncl', default=3000, type=int, metavar='INT', help='number of clusters')
    parser.add_argument('--hc', default=10, type=int, metavar='INT', help='number of heads')

    # optimization
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=146, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.1, type=float, metavar='FLOAT', help='initial learning rate')

    # other
    parser.add_argument('--ckpt-dir', default='.test', metavar='DIR', help='path to result dirs')
    parser.add_argument('--datadir', default='/home/ubuntu/data/imagenet', type=str,help='')
    parser.add_argument('--modelpath', default='./checkpoint999.pth', type=str,help='')
    parser.add_argument('--name', default='test', type=str, help='comment for tensorboardX')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate only')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    # Setup CUDA and random seeds
    util.setup_runtime(seed=42, cuda_dev_id=args.device)
    model = data.return_model_loader(args, return_loader=False)
    util.prepmodel(model, args.modelpath)

    name = "%s" % args.name.replace('/', '_')
    writer = SummaryWriter('./eval/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))
    # Setup dataset
    train_loader, val_loader = data.get_standard_data_loader_pairs(dir_path=args.datadir,
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.workers)
    print("LENDATA:", len(train_loader.dataset), flush=True)
    # Setup optimizer
    o = StandardOptimizer(weight_decay=0)
    def lr_schedule(epoch):
        if epoch < 85:
            return args.learning_rate
        elif epoch < 128:
            return args.learning_rate/10.
        else:
            return args.learning_rate/100.
    print(model.top_layer, flush=True)
    o.lr_schedule = lambda epoch:  lr_schedule(epoch)
    o.writer = writer
    o.resume = True
    o.lr = args.learning_rate
    o.validate_only = args.evaluate
    o.num_epochs = args.epochs
    o.checkpoint_dir = args.ckpt_dir

    # Optimize
    o.optimize(model, train_loader, val_loader)

