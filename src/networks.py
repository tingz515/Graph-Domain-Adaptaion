import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HyperLinear(nn.Module):
    def __init__(
        self,
        ndomains: int,
        embedding_dim: int,
        hidden_dim: int,
        hidden_num: int,
        feature_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        input_dim = ndomains
        output_dim = feature_dim * output_dim + output_dim
        # self.embed = nn.Embedding(input_dim, embedding_dim)
        self.embed = nn.Sequential(nn.Embedding(input_dim, embedding_dim), nn.ReLU())
        # Do it needs ReLU after Embedding ?
        dims = [embedding_dim] + [hidden_dim] * hidden_num
        model = []
        for i in range(len(dims) - 1):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU())
        model.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*model)

        self.in_features = feature_dim
        self.out_features = output_dim

    def base_forward(self, x, param):
        weight, bias = torch.split(
            param, [self.feature_dim * self.output_dim, self.output_dim], dim=-1
        )
        weight = weight.reshape(self.output_dim, self.feature_dim)
        bias = bias.reshape(self.output_dim)
        out = F.linear(x, weight, bias)
        # weight = weight.reshape(-1, self.output_dim, self.feature_dim)
        # bias = bias.reshape(-1, self.output_dim)
        # out = torch.einsum("bn, bon -> bo", x, weight) + bias
        return out

    def forward(self, x, id):
        embed = self.embed(id)
        param = self.model(embed)
        if self.feature_dim == 0:
            return param
        out = self.base_forward(x, param)
        return out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv2d') or classname.startswith('ConvTranspose2d'):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.startswith('BatchNorm'):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.startswith('Linear'):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim, device):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1. / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, ndomains):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)  # disable for digits
    self.ad_layer3 = nn.Linear(hidden_size, ndomains)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()  # disable for digits
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)  # disable for digits
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    self.ndomains = ndomains

  def output_num(self):
    return self.ndomains

  def get_parameters(self):
    return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

  def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
      return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

  def grl_hook(self, coeff):
      def fun1(grad):
          return -coeff * grad.clone()

      return fun1

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = self.calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    if self.training:
        x.register_hook(self.grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)  # disable for digits
    x = self.relu2(x)  # disable for digits
    x = self.dropout2(x)  # disable for digits
    y = self.ad_layer3(x)
    return y


resnet_dict = {'ResNet18': models.resnet18, 'ResNet34': models.resnet34, 'ResNet50': models.resnet50,
               'ResNet101': models.resnet101, 'ResNet152': models.resnet152}


class ResNetFc(nn.Module):
    def __init__(
        self,
        resnet_name,
        use_bottleneck=True,
        bottleneck_dim=256,
        new_cls=False,
        class_num=1000,
        domain_num=3,
        hyper_embed_dim=64,
        hyper_hidden_dim=128,
        hyper_hidden_num=1,
        use_hyper=True
    ):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
                                            self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.use_hyper = use_hyper
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.apply(init_weights)
                if use_hyper:
                    self.fc = HyperLinear(domain_num, hyper_embed_dim, hyper_hidden_dim, hyper_hidden_num, bottleneck_dim, class_num)
                else:
                    self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, id=None):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
            x = F.relu(x)
        if self.use_hyper:
            y = self.fc(x, id)
        else:
            y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
            else:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
        else:
            parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list
