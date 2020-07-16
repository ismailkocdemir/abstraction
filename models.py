import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np 
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = [
    "cnn5", 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19_bn_lb', 'vgg19_bn_4','vgg19_bn_8','vgg19_bn_16','vgg19_bn_32',
    'vgg19', 'vgg19_4','vgg19_8','vgg19_16','vgg19_32',
    'resnet18k_4','resnet18k_8','resnet18k_16','resnet18k_32','resnet18preAct',
    'resnet18', 'resnet34', 'resnet50', 'resnet101'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class CNN5(nn.Module):
    def __init__(self, features):
        super(CNN5, self).__init__()
        self.features = features
    
    def forward(self, x): 
        return self.features(x)

    def print_arch(self):
        for name, param in self.named_parameters():
            print(name, param.size())

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

def make_cnn(c=64):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*8, 10, bias=True)
    )


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, width=64, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.width = int(512 * (width/64.0))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(True),
            nn.Linear(self.width, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x,):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def update_reference_weights(self,):
        for name, param in self.named_parameters():
            if name.split(".")[-1] == "bias" or (len(param.size()) == 1 and name.split(".")[0] == "features"):
                continue
            buffer_key = name.replace(".", "_")
            self._buffers[buffer_key] = param.detach().cpu()


    def print_arch(self):
        for name, param in self.named_modules():
            print(name, param)

    def get_cosine_distance_per_layer(self):
        param_keys = []
        distances = []
        for name, param in self.named_parameters():
            if (name.split(".")[-1] == "bias") or (len(param.size()) == 1 and name.split(".")[0] == "features"):
                continue

            buffer_key = name.replace(".", "_")

            initial_version = self._buffers[buffer_key].view(-1, np.prod(list(param.size())) )
            param_flattened = param.detach().view(-1, np.prod( list(param.size())))
            distance = 1 - F.cosine_similarity(param_flattened, initial_version.to(device))
            distances.append(distance.detach().cpu().numpy()[0])
            param_keys.append(name)

        return param_keys, distances

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False, width=64):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            curr_v = int(width*(v/64.0))
            conv2d = nn.Conv2d(in_channels, curr_v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(curr_v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = curr_v
    return nn.Sequential(*layers)

def cnn5(c=5):
    """5 layer CNN model (identical with model from Double descent paper.)"""
    return CNN5(make_cnn(c))

def vgg11(num_classes):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes=num_classes)


def vgg11_bn(num_classes):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


def vgg13(num_classes):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes=num_classes)


def vgg13_bn(num_classes):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)


def vgg16(num_classes):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_classes)


def vgg16_bn(num_classes):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)


def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes)

def vgg19_4(num_classes):
    """VGG 19-layer model (configuration 'E')"""
    return VGG(make_layers(cfg['E'], width=4),width=4, num_classes=num_classes)


def vgg19_8(num_classes):
    """VGG 19-layer model (configuration 'E')"""
    return VGG(make_layers(cfg['E'], width=8),width=8, num_classes=num_classes)


def vgg19_16(num_classes):
    """VGG 19-layer model (configuration 'E')"""
    return VGG(make_layers(cfg['E'], width=16),width=16, num_classes=num_classes)


def vgg19_32(num_classes):
    """VGG 19-layer model (configuration 'E')"""
    return VGG(make_layers(cfg['E'], width=32),width=32, num_classes=num_classes)



def vgg19_bn(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)

def vgg19_bn_lb(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)


def vgg19_bn_4(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, width=4),width=4, num_classes=num_classes)


def vgg19_bn_8(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, width=8),width=8, num_classes=num_classes)


def vgg19_bn_16(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, width=16),width=16, num_classes=num_classes)


def vgg19_bn_32(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, width=32),width=32, num_classes=num_classes)




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks,  init_channels=64, num_classes = 10):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels
        

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.fc = nn.Linear(8*c*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Save initial wegihts in buffers
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:
                self.register_buffer(name.replace(".", "_"), param.detach().cpu())
                #print("buffer {0} is registered".format(name.replace(".", "_")))


    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def print_arch(self):
        for name, param in self.named_parameters():
            print(name, param.size())

    
    def update_reference_weights(self,):
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:
                buffer_key = name.replace(".", "_") #"_".join(splt[:3])
                self._buffers[buffer_key] = param.detach().cpu()

    def get_cosine_distance_per_layer(self):
        param_keys = []
        distances = []
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:

                buffer_key = name.replace(".", "_") #"_".join(splt[:3])

                initial_version = self._buffers[buffer_key].view(-1, np.prod(list(param.size())) ).to(device)
                param_flattened = param.detach().view(-1, np.prod( list(param.size())))
                distance = 1- F.cosine_similarity(param_flattened, initial_version)
                distances.append(distance.detach().cpu().numpy()[0])
                param_keys.append(name)

        return param_keys, distances

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Save initial wegihts in buffers
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:
                self.register_buffer(name.replace(".", "_"), param.detach().cpu())
                #print("buffer {0} is registered".format(name.replace(".", "_")))

    def update_reference_weights(self,):
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:
                buffer_key = name.replace(".", "_") #"_".join(splt[:3])
                self._buffers[buffer_key] = param.detach().cpu()

    def get_cosine_distance_per_layer(self):
        param_keys = []
        distances = []
        for name, param in self.named_parameters():
            if "conv" in name or "fc.weight" in name:
                #buffer_key = None
                #splt = name.split(".")
                buffer_key = name.replace(".", "_") #"_".join(splt[:3])

                initial_version = self._buffers[buffer_key].view(-1, np.prod(list(param.size())) ).to(device)
                param_flattened = param.detach().view(-1, np.prod( list(param.size())))
                distance = 1- F.cosine_similarity(param_flattened, initial_version)
                distances.append(distance.detach().cpu().numpy()[0])
                param_keys.append(name)

        return param_keys, distances


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


    # Allow for accessing forward method in a inherited class
    forward = _forward

    def print_arch(self):
        for name, param in self.named_parameters():
            print(name, param.size())



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
    

def resnet18k_4(num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], init_channels=4, num_classes=num_classes)

def resnet18k_8(num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], init_channels=8, num_classes=num_classes)

def resnet18k_16(num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], init_channels=16, num_classes=num_classes)

def resnet18k_32(num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], init_channels=32, num_classes=num_classes)

def resnet18preact(num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], init_channels=64, num_classes=num_classes)

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
