# adapted from DeepCluster repo: https://github.com/facebookresearch/deepcluster
import math
import torch.nn as nn

__all__ = [ 'AlexNet', 'alexnet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    'big': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'],
    'small': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, num_classes, init=True):
        super(AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True))
        self.headcount = len(num_classes)
        self.return_features = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(4096, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])#,bias=False)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet(bn=True, num_classes=[1000], init=True, size='big'):
    dim = 3
    model = AlexNet(make_layers_features(CFG[size], dim, bn=bn), num_classes, init)
    return model

if __name__ == '__main__':
    import torch
    model = alexnet(num_classes=[500]*3)
    print([ k.shape for k in model(torch.randn(64,3,224,224))])
