import torch
import torchvision

from util import setup_runtime
import models

setup_runtime(42, [2])
ARCH = 'resnet50'
HC = 1
K = 3000
m = 'path-model_name_with_old_style.pth'
name = 'path-model_name_that_loads_as_torchvision_resnet.pth'
###########################################################################
if ARCH =='resnet50':
    model = models.resnet50(out=[K]*HC)
    base = torchvision.models.resnet50(pretrained=False, num_classes=K)
else:
    model = models.resnet18(out=[K]*HC)
    base = torchvision.models.resnet18(pretrained=False, num_classes=K)

ckpt = torch.load(m, map_location='cpu')

new_ckpt = {}
changer = {'features.0':'conv1',
           'features.1':'bn1',
           'features.4':'layer1', 'features.5':'layer2', 'features.6':'layer3', 'features.7':'layer4',
           'top_layer':'fc'
           }
for k,v in ckpt['state_dict'].items():
    k2 = k
    for orig, repl in changer.items():
        k2 = k2.replace(orig, repl)
    new_ckpt[k2] = v
print(new_ckpt.keys())
if HC > 1:
    new_ckpt['fc.weight'] = new_ckpt['fc0.weight']
    new_ckpt['fc.bias'] = new_ckpt['fc0.bias']
    del new_ckpt['fc0.bias'], new_ckpt['fc0.weight']

new_ckpt = {k:v for k,v in new_ckpt.items() if k in base.state_dict().keys()}

base.load_state_dict(new_ckpt)
print('loaded successfully')
torch.save({'state_dict': new_ckpt, 'labels': ckpt['L'].cpu()}, f'{name}.pth')