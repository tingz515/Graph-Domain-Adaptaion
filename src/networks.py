from collections import OrderedDict
from torchvision import models

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG = False

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
        self.in_features = feature_dim
        self.out_features = output_dim

        # self.embed = nn.Embedding(ndomains, embedding_dim)
        embed = torch.normal(0, 1, size=(ndomains, embedding_dim))
        embed /= torch.norm(embed, dim=-1, keepdim=True)
        self.embed = nn.Parameter(embed, requires_grad=True)
        # self.embed = nn.Parameter(torch.normal(0, 1, size=(ndomains, embedding_dim)), requires_grad=True)
        # nn.init.xavier_normal_(self.embed)
        # Do it needs ReLU after Embedding ?
        output_dim = feature_dim * output_dim + output_dim
        dims = [embedding_dim] + [hidden_dim] * hidden_num
        model = []
        for i in range(len(dims) - 1):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU())
        model.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*model)

    def base_forward(self, x, param):
        weight, bias = torch.split(
            param, [self.in_features * self.out_features, self.out_features], dim=-1
        )
        weight = weight.reshape(self.out_features, self.in_features)
        bias = bias.reshape(self.out_features)
        out = F.linear(x, weight, bias)
        # weight = weight.reshape(-1, self.out_features, self.in_features)
        # bias = bias.reshape(-1, self.out_features)
        # out = torch.einsum("bn, bon -> bo", x, weight) + bias
        return out

    def forward(self, x, id):
        embed = F.normalize(self.embed)[id]
        # embed = self.embed[id]
        embed = F.relu(embed)
        param = self.model(embed)
        if self.in_features == 0:
            return param
        out = self.base_forward(x, param)
        return out

    def get_param(self, id):
        embed = F.normalize(self.embed)[id]
        # embed = self.embed[id]
        embed = F.relu(embed)
        param = self.model(embed)
        weight, bias = torch.split(
            param, [self.in_features * self.out_features, self.out_features], dim=-1
        )
        weight = weight.reshape(self.out_features, self.in_features)
        bias = bias.reshape(self.out_features)
        return OrderedDict({"weight": weight, "bias": bias})



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
        use_hyper=False,
        multi_mlp=False,
        prompt_num=0,
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
        if DEBUG:
            self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.avgpool)
        else:
            self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
                                                self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.use_hyper = use_hyper
        self.multi_mlp = multi_mlp
        if new_cls:
            self.__in_features = model_resnet.fc.in_features
            if self.use_bottleneck:
                self.__in_features = bottleneck_dim
                if DEBUG:
                    self.bottleneck = nn.Linear(256, bottleneck_dim)
                else:
                    self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.apply(init_weights)
            if use_hyper:
                self.fc = HyperLinear(domain_num, hyper_embed_dim, hyper_hidden_dim, hyper_hidden_num, self.__in_features, class_num)
            elif multi_mlp:
                fc = [nn.Linear(self.__in_features, class_num) for _ in range(domain_num)]
                self.fc = nn.ModuleList(fc)
            else:
                self.fc = nn.Linear(self.__in_features, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

        self.prompt_num = prompt_num
        if prompt_num > 0:
            x_prompt = torch.normal(0, 1, size=(prompt_num, 1, 3, 224, 224))
            self.x_prompt = nn.Parameter(x_prompt, requires_grad=True)

        self._init_light_model()

    def _init_light_model(self):
        resnet20 = resnet_dict['ResNet18'](pretrained=True)
        conv1 = resnet20.conv1
        bn1 = resnet20.bn1
        relu = resnet20.relu
        maxpool = resnet20.maxpool
        layer1 = resnet20.layer1
        layer2 = resnet20.layer2
        layer3 = resnet20.layer3
        layer4 = resnet20.layer4
        avgpool = resnet20.avgpool

        self.light_model = nn.Sequential(conv1, bn1, relu, maxpool, layer1,
                                            layer2, layer3, layer4, avgpool)
        self.light_bottleneck = nn.Linear(resnet20.fc.in_features, self.bottleneck.out_features)
        self.light_bottleneck.apply(init_weights)

    def light_feature(self, x):
        x = self.light_model(x)
        x = x.view(x.size(0), -1)
        x = self.light_bottleneck(x)
        x = F.relu(x)
        return x

    def light_forward(self, x, id=None):
        x = self.light_feature(x)
        if self.use_hyper:
            y = self.fc(x, id)
        elif self.multi_mlp:
            y = self.fc[id](x)
        else:
            y = self.fc(x)
        return x, y

    def forward(self, x, id=None):
        if self.prompt_num > 1:
            x = x + self.x_prompt[id]
        elif self.prompt_num > 0:
            x = x + self.x_prompt[0]
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
            x = F.relu(x)
        if self.use_hyper:
            y = self.fc(x, id)
        elif self.multi_mlp:
            y = self.fc[id](x)
        else:
            y = self.fc(x)
        return x, y

    def progressive_forward(self, x, id=None):
        if self.prompt_num > 1:
            x = x + self.x_prompt[id]
        elif self.prompt_num > 0:
            x = x + self.x_prompt[0]
        x_s = self.large_feature(x)
        x_t = self.light_feature(x)
        if self.use_hyper:
            y_t = self.fc(x_t, id)
            y_s = self.fc(x_s, 0)
        elif self.multi_mlp:
            y_t = self.fc[id](x_t)
            y_s = self.fc[0](x_s)
        return x_t, y_t, y_s

    def large_feature(self, x):
        if self.prompt_num > 0:
            x = x + self.x_prompt[0]
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
            x = F.relu(x)
        return x

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
                if self.prompt_num > 0:
                    parameter_list.append({'params': self.x_prompt, 'lr_mult': 10, 'decay_mult': 2})
            else:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
        else:
            parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        parameter_list.extend([
            {'params': self.light_model.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.light_bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}
            ])
        return parameter_list

    def get_fc_parameters(self):
        if self.use_hyper:
            parameter_list = [
                {'params': self.fc.embed, 'lr_mult': 10, 'decay_mult': 2}
            ]
            if self.prompt_num > 1:
                parameter_list.append({'params': self.x_prompt, 'lr_mult': 10, 'decay_mult': 2})
        else:
            parameter_list = [
                {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
            ]
        parameter_list.extend([
            {'params': self.light_model.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.light_bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}
            ])
        return parameter_list
