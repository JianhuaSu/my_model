import torch
import math
from torch import nn
from torch.nn.parameter import Parameter
from .BMSZ import BMSZ

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale = 32, device = None):

        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).to(device))
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

class MLP_head(nn.Module):
    
    def __init__(self, args, num_classes):
        
        super(MLP_head, self).__init__()
        self.args = args
        self.num_classes = num_classes

        if num_classes == 2:
            self.layer1 = nn.Linear(args.hidden_dims, args.mlp_hidden_size)
            self.layer2 = nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size)   
            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(p=args.mlp_dropout)
            self.output_layer = nn.Linear(args.mlp_hidden_size, num_classes)

        else:
            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            if self.args.ablation == 'no_CosClass':
                self.output_layer_1 = nn.Linear(args.hidden_dims, args.num_labels)
            else:
                self.output_layer_1 = CosNorm_Classifier(args.hidden_dims, args.num_labels, args.scale, args.device)


    def adjust_scores(self, scores):
        adjusted_scores = scores / (1 - scores)
        return adjusted_scores

    def forward(self, x, binary_scores = None):

        if self.num_classes == 2:
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.output_layer(x)
            
            return x

        else:
            if self.args.ablation != 'no_binary':
                binary_scores = self.adjust_scores(binary_scores)
                binary_scores = binary_scores.unsqueeze(1).expand(-1, x.shape[1])
                fusion_x = x * binary_scores
                logits = self.output_layer_1(fusion_x)
            else:
                logits = self.output_layer_1(x)

            return logits

    
    def vim(self):
        
        w = self.output_layer_1.weight
        b = torch.zeros(w.size(0))
       
        return w, b

class MMEncoder(nn.Module):

    def __init__(self, args):

        super(MMEncoder, self).__init__()
        self.model = BMSZ(args)

    def forward(self, text_feats, video_feats, audio_feats, labels = None, ood_sampling = False):
        
        pooled_output, mixed_labels = self.model(text_feats, video_feats, audio_feats, labels, ood_sampling = ood_sampling)
        return pooled_output, mixed_labels

        
 
