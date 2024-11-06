from torch.nn.parameter import Parameter
from torch import nn
import math
import torch


class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale = 32):

        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = Parameter(torch.Tensor(out_dims, in_dims))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):

        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = input / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)

        return torch.mm(ex * self.scale, ew.t())