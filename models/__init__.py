import torch
import torchvision
import torch.nn.functional as F

from .synthesizer_net import InnerProd, Bias
from .audio_net import Unet
from .vision_net import ResnetFC, ResnetDilated
from .criterion import *
from .dprnn import DPRNN_TasNet

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.0001)


def build_sound(arch='unet5', fc_dim=64, weights=''):
    # 2D models
    if arch == 'unet5':
        net_sound = Unet(fc_dim=fc_dim, num_downs=5)
    elif arch == 'unet6':
        net_sound = Unet(fc_dim=fc_dim, num_downs=6)
    elif arch == 'unet7':
        net_sound = Unet(fc_dim=fc_dim, num_downs=7)
    elif arch == 'dprnn6':
        net_sound = DPRNN_TasNet(num_spk=1)
    else: raise Exception('No such arch-sound')

    net_sound.apply(weights_init)
    return net_sound

# builder for vision
def build_frame(arch='resnet18', fc_dim=64, pool_type='avgpool',
                weights=''):
    pretrained=True
    if arch == 'resnet18fc':
        original_resnet = torchvision.models.resnet18(pretrained)
        net = ResnetFC(
            original_resnet, fc_dim=fc_dim, pool_type=pool_type)
    elif arch == 'resnet18dilated':
        original_resnet = torchvision.models.resnet18(pretrained)
        net = ResnetDilated(
            original_resnet, fc_dim=fc_dim, pool_type=pool_type)
    else:
        raise Exception('Architecture undefined!')

    return net

def build_synthesizer(arch, fc_dim=64, weights=''):
    if arch == 'linear':
        net = InnerProd(fc_dim=fc_dim)
    elif arch == 'bias':
        net = Bias()
    else:
        raise Exception('Architecture undefined!')

    net.apply(weights_init)
    return net