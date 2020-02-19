import os
import glob
import torch


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    if path is not None and not os.path.exists(path):
        os.makedirs(path)

def get_model_device(model):
    return next(model.parameters()).device

def save_model(model, model_path):
    """Save the model state dictionary to the PTH file model_path."""
    if model_path is not None:
        xmkdir(os.path.dirname(model_path))
        torch.save(model.state_dict(), model_path)

def load_checkpoint(checkpoint_dir, model, optimizer=None):
    """Search the latest checkpoint in checkpoint_dir and load the model and optimizer and return the metrics."""
    names = list(sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
    ))
    if len(names) == 0:
        return 0, {'train': [], 'val': []}
    print(f"Loading checkpoint '{names[-1]}'")
    cp = torch.load(names[-1], map_location=str(get_model_device(model)))
    epoch = cp['epoch']
    metrics = cp['metrics']
    if model:
        model.load_state_dict(cp['model'])
    if optimizer:
        optimizer.load_state_dict(cp['optimizer'])
    return epoch, metrics


def clean_checkpoint(checkpoint_dir, lowest=False):
    if lowest:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'lowest*.pth'))
        ))
    else:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
    if len(names) > 2:
        for name in names[0:-2]:
            print(f"Deleting redundant checkpoint file {name}")
            os.remove(name)


def save_checkpoint(checkpoint_dir, model, optimizer, metrics, epoch, defsave=False):
    """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir
    for the specified epoch. If checkpoint_dir is None it does not do anything."""
    if checkpoint_dir is not None:
        if model:
            xmkdir(checkpoint_dir)
            if (epoch % 50 == 0) or defsave:
                name = os.path.join(checkpoint_dir, f'ckpt{epoch:08}.pth')
            else:
                name = os.path.join(checkpoint_dir, f'checkpoint{epoch:08}.pth')
            torch.save({
                'epoch': epoch + 1,
                'metrics': metrics,
                'model': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, name)
            clean_checkpoint(checkpoint_dir)
        else:
            xmkdir(checkpoint_dir)
            if (epoch % 50 == 0) or defsave:
                name = os.path.join(checkpoint_dir, f'ckpt{epoch:08}.pth')
            else:
                name = os.path.join(checkpoint_dir, f'checkpoint{epoch:08}.pth')
            torch.save({
                'epoch': epoch + 1,
                'metrics': metrics,
                'optimizer': optimizer.state_dict(),
            }, name)
            clean_checkpoint(checkpoint_dir)


def save_checkpoint_all(checkpoint_dir, model, arch, opt, L, epoch, lowest=False, save_str=''):
    """Save model, optimizer, and metrics state to a checkpoint in
    checkpoint_dir for the specified epoch. If checkpoint_dir is None it does not do anything."""
    if checkpoint_dir is not None:
        if model:
            xmkdir(checkpoint_dir)
            if (epoch % 50 == 0) or (save_str != ''):
                name = os.path.join(checkpoint_dir, f'ckpt{epoch:03}.pth')
            else:
                name = os.path.join(checkpoint_dir, f'checkpoint{epoch:03}.pth')
            if lowest:
                name = os.path.join(checkpoint_dir, f'lowest_{epoch:03}.pth')
            torch.save({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'L': L,
            }, name)
            clean_checkpoint(checkpoint_dir, lowest=lowest)
        else:
            xmkdir(checkpoint_dir)
            if epoch % 50 == 0:
                name = os.path.join(checkpoint_dir, f'ckpt{epoch:03}.pth')
            else:
                name = os.path.join(checkpoint_dir, f'checkpoint{epoch:03}.pth')
            if lowest:
                name = os.path.join(checkpoint_dir, f'lowest.pth')
            torch.save({
                'epoch': epoch + 1,
                'arch': arch,
                'optimizer': opt.state_dict(),
                'L': L,
            }, name)
            clean_checkpoint(checkpoint_dir, lowest=lowest)


def load_checkpoint_all(checkpoint_dir, model, opt):
    """Search the latest checkpoint in checkpoint_dir and load the model and optimizer and return the metrics."""
    names = list(sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
    ))
    if len(names) == 0:
        return [], 0
    print(f"Loading checkpoint '{names[-1]}'")
    if model:
        cp = torch.load(names[-1], map_location=str(get_model_device(model)))
    else:
        cp = torch.load(names[-1], map_location=str('cpu'))
    epoch = cp['epoch']
    if opt:
        opt.load_state_dict(cp['optimizer'])
    L = cp['L']

    if model:
        model_parallel = 'module' in list(model.state_dict().keys())[0]
        if ('module' in list(cp['state_dict'].keys())[0]) and not model_parallel:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in cp['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print('loaded from parallel to single!',flush=True)
        else:
            model.load_state_dict(cp['state_dict'])

    return L, epoch
