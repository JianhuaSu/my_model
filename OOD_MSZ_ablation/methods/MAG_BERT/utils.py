import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from losses import loss_map
from backbones.SubNets.transformers_encoder.transformer import TransformerEncoder
from backbones.SubNets.FeatureNets import BERTEncoder
from torch.autograd import Variable
from backbones.SubNets.AlignNets import AlignSubNet
from transformers import BertTokenizer

class BinaryClassificationLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassificationLayer, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 64)
        self.layer3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x_last = self.dropout(x)
        x = self.layer3(x_last)
        return x, x_last

class MMClassificationLayer(nn.Module):
    def __init__(self, args, input_size=768):
        super(MMClassificationLayer, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, args.num_labels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x, input_size, device):
        self.layer1 = nn.Linear(input_size, 128).to(device)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x_last = self.dropout(x)
        logits = self.layer3(x_last)
        return logits ,x_last

def get_transformer_encoder(args, embed_dim, layers):

    return TransformerEncoder(embed_dim=embed_dim,
                                num_heads=args.b_nheads,
                                layers=layers,
                                attn_dropout=args.b_attn_dropout,
                                relu_dropout=args.b_relu_dropout,
                                res_dropout=args.b_res_dropout,
                                embed_dropout=args.b_embed_dropout,
                                attn_mask=args.b_attn_mask
                            )

def _random_token_erase(self, input_ids):

        masked_ids = []
        for inp_i in input_ids:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_i, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            masked_inds = np.random.choice(inds, size = int(len(inds) * self.args.re_prob), replace = False)
            sent_masked_inds = sent_tokens_inds[masked_inds]
            masked_ids.append(list(sent_masked_inds))

        return masked_ids

class TextEncoder(nn.Module):

    def __init__(self, args):

        super(TextEncoder, self).__init__()

        self.args = args
        self.text_embedding = BERTEncoder(args)

        for param in self.text_embedding.parameters():
            param.requires_grad = False

        self.encoder = get_transformer_encoder(args, args.text_feat_dim, args.b_encoder_layers_1)
        self.text_binary_classifier = BinaryClassificationLayer(args.text_feat_dim, 128)

    def forward(self, text_feats, labels = None, ood_sampler = None, device = None, select_elems = None, mode = 'eval'):
        
        # if mode == 'pretrain':
        #     text = self.text_embedding(text_feats)
        # else:
        #     text = text_feats

        text = self.text_embedding(text_feats)
        # print("======",text.shape)
        # print("+++++++",text)
        if ood_sampler is not None:
            # print('mode:', mode)
            mix_feats, mix_labels = ood_sampler(text, labels, mode = mode, device = device)
            text = mix_feats
            labels = mix_labels['binary']
        
        if select_elems is not None:

            if len(select_elems) > 0:

                idx_a = [e[0] for e in select_elems]
                idx_b = [e[2] for e in select_elems]
                alphas = [e[1] for e in select_elems]

                ood_feats = []
                ood_len = text.shape[0]

                for i in range(len(idx_a)):
                    feat_a = text[idx_a[i]]
                    feat_b = text[idx_b[i]]
                    alpha = alphas[i]
                    ood_feat = feat_a * alpha + (1 - alpha) * feat_b
                    # print('1111111', ood_feat.shape)
                    ood_feats.append(ood_feat)
                
                # print('11111', ood_len)
                # print('222222', text.shape[1])
                # print('3333333', len(ood_feats))

                ood_feats = torch.cat(ood_feats, dim = 0).view(ood_len, text.shape[1], -1)
                # print(f"shape ood feats: {ood_feats.shape} shape text: {text.shape}")
                text = torch.cat((text, ood_feats), dim = 0)  

        text_per = text.permute(1, 0, 2)
        text = self.encoder(text_per)[-1]

        # text = text[:, 0]
        # print('1111111', text.shape)
        text_logits, h = self.text_binary_classifier(text)

        outputs = {
            'mm': text_logits,
            'h': h,
            'labels': labels,
            'feats' : text
        }

        return outputs

class MMEncoder(nn.Module):

    def __init__(self, args, feat_dim):

        super(MMEncoder, self).__init__()
        self.encoder = get_transformer_encoder(args, feat_dim, args.b_encoder_layers_1)
        self.binary_classifier = BinaryClassificationLayer(feat_dim, 128)
    
    def forward(self, x, labels = None, ood_sampler = None, device = None, select_elems = None, mode = 'eval'):
        
        x = x.float()

        if ood_sampler is not None:

            mix_feats, mix_labels = ood_sampler(x, labels, mode = mode, device = device)
            x = mix_feats
            labels = mix_labels['binary']

        if select_elems is not None:

            if len(select_elems) > 0:

                idx_a = [e[0] for e in select_elems]
                idx_b = [e[2] for e in select_elems]
                alphas = [e[1] for e in select_elems]

                ood_feats = []
                ood_len = x.shape[0]

                for i in range(len(idx_a)):
                    feat_a = x[idx_a[i]]
                    feat_b = x[idx_b[i]]
                    alpha = alphas[i]
                    ood_feat = feat_a * alpha + (1 - alpha) * feat_b
                    ood_feats.append(ood_feat)

                ood_feats = torch.cat(ood_feats, dim = 0).view(ood_len, x.shape[1], -1)
                x = torch.cat((x, ood_feats), dim = 0)  

        x_per = x.permute(1, 0, 2)
        x_per = self.encoder(x_per)[-1]
        s_x, s_x_last = self.binary_classifier(x_per)

        outputs = {
            'mm': s_x,
            'h': s_x_last,
            'labels': labels,
            'feats' : x_per
        }
        return outputs

class view_generator:
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=self.args.rtr_prob)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos
    
    def random_token_erase(self, input_x, input_mask, max_seq_length, mode = 'text', re_prob = 0.25):
        
        aug_input_x = []
        aug_input_mask = []
        
        for inp_x, inp_m in zip(input_x, input_mask):
            
            if mode == 'text':
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_x, already_has_special_tokens=True)
                sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
                
                inds = np.arange(len(sent_tokens_inds))
                masked_inds = np.random.choice(inds, size = int(len(inds) * re_prob), replace = False)
                sent_masked_inds = sent_tokens_inds[masked_inds]

                inp_x = np.delete(inp_x, sent_masked_inds)
                inp_x = F.pad(inp_x, (0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)
                
                inp_m = np.delete(inp_m, sent_masked_inds)
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)
            else:
                sent_tokens_inds = np.where(inp_m.numpy() == 1)[0]

                erase_start_ind = np.random.choice(sent_tokens_inds)
                erase_end_ind = min(erase_start_ind + max(1, int(re_prob * (len(sent_tokens_inds) - erase_start_ind))), len(sent_tokens_inds))
                erase_inds = sent_tokens_inds[erase_start_ind: erase_end_ind]

                inp_x = np.delete(inp_x, erase_inds, axis = 0)
                inp_x = F.pad(inp_x, (0, 0, 0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)

                inp_m = np.delete(inp_m, erase_inds)
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)

            aug_input_x.append(inp_x)
            aug_input_mask.append(inp_m)
        
        aug_input_x = torch.stack(aug_input_x, dim=0)
        aug_input_mask = torch.stack(aug_input_mask, dim=0)
        
        return aug_input_x, aug_input_mask

class MMOODSampler(nn.Module):
    
    def __init__(self, args):
        super(MMOODSampler, self).__init__()
        self.ood_label_id = args.ood_label_id
        self.args = args
        
    def forward(self, feats, label_ids, mode = 'pretrain', device = None):
        
        alpha = self.args.b_alpha if mode == 'pretrain' else self.args.alpha

        num_ood = len(feats) * self.args.multiple_ood
        
        ood_list = []
        
        seq_length = feats.shape[1]    

        numpy_labels = label_ids.cpu().numpy()
        # print('0000000', numpy_labels)
        unique_labels = np.unique(numpy_labels)
        # print('00000000', unique_labels)

        if len(unique_labels) > 1:
            while len(ood_list) < num_ood:
                
                select_labels = np.random.choice(unique_labels, 2, replace=False)
                # print('1111111', select_labels)

                select_label_a = select_labels[0]
                select_label_b = select_labels[1]
                # print('222222, a:{}, b:{}'.format(select_label_a, select_label_b))

                all_pos_a = list(np.where(numpy_labels == select_label_a))[0]
                all_pos_b = list(np.where(numpy_labels == select_label_b))[0]
                # print('222222, a:{}, b:{}'.format(all_pos_a, all_pos_b))

                pos_a = np.random.choice(np.array(all_pos_a), 1)
                pos_b = np.random.choice(np.array(all_pos_b), 1)
                # print('222222, a:{}, b:{}'.format(pos_a, pos_b))

                s = np.random.beta(alpha, alpha)  
                # s = np.random.uniform(0, 1, 1)[0]
                
                ood_list.append(s * feats[pos_a] + (1 - s) * feats[pos_b])

            if feats.ndim == 3:
                ood_feats = torch.cat(ood_list, dim = 0).view(num_ood, seq_length, -1)
            
            elif feats.ndim == 2:
                ood_feats = torch.cat(ood_list, dim = 0).view(num_ood, -1)

            mix_feats = torch.cat((feats, ood_feats), dim = 0)
            semi_label_ids = torch.cat((label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
            binary_label_ids = torch.cat((torch.tensor([1] * len(feats)) , torch.tensor([0] * num_ood)), dim=0)
        
        else:
            mix_feats = feats
            semi_label_ids = label_ids
            binary_label_ids = torch.tensor([1] * len(feats))

        mix_feats = mix_feats.to(device)

        mix_labels = {
            'semi': semi_label_ids.to(device),
            'binary': binary_label_ids.to(device)
        }
       
        return mix_feats, mix_labels 

def get_select_elems(alpha, num_id, num_ood, labels):

    cnt = 0
    select_elems = []
    numpy_labels = labels.cpu().numpy()
    unique_labels = np.unique(numpy_labels)

    if len(unique_labels) > 1:
        while cnt < num_ood:

            select_labels = np.random.choice(unique_labels, 2, replace=False)

            select_label_a = select_labels[0]
            select_label_b = select_labels[1]

            all_pos_a = list(np.where(numpy_labels == select_label_a))[0]
            all_pos_b = list(np.where(numpy_labels == select_label_b))[0]

            pos_a = np.random.choice(np.array(all_pos_a), 1)
            pos_b = np.random.choice(np.array(all_pos_b), 1)

            s = np.random.beta(alpha, alpha)  
            cnt += 1
            select_elems.append([pos_a, s, pos_b])

    return select_elems

# class TOODSampler(nn.Module):
    
#     def __init__(self, args):
#         super(MMOODSampler, self).__init__()
#         self.ood_label_id = args.ood_label_id
#         self.args = args
        
#     def forward(self, ind_text_feats, ind_video_data, ind_audio_data, ind_label_ids, device = None):
        
#         num_ood = len(ind_text_feats) * self.args.multiple_ood
        
#         ood_text_list, ood_video_list, ood_audio_list = [], [], []
        
#         text_seq_length, video_seq_length, audio_seq_length = \
#             ind_text_feats.shape[1], ind_video_data.shape[1], ind_audio_data.shape[1]
            
#         select_elems = []

#         while len(ood_text_list) < num_ood:
#             cdt = np.random.choice(ind_label_ids.size(0), 2, replace=False)
            
#             if ind_label_ids[cdt[0]] != ind_label_ids[cdt[1]]:
                
              
#                 alpha = 2
#                 s = np.random.beta(alpha, alpha)  
#                 # print('s:', s)
#                 # s = np.random.uniform(0, 1, 1)[0]
                
#                 ood_text_list.append(s * ind_text_feats[cdt[0]] + (1 - s) * ind_text_feats[cdt[1]])
#                 ood_video_list.append(s * ind_video_data[cdt[0]] + (1 - s) * ind_video_data[cdt[1]])
#                 ood_audio_list.append(s * ind_audio_data[cdt[0]] + (1 - s) * ind_audio_data[cdt[1]])

#                 select_elems.append([cdt[0], s, cdt[1]])

#         if ind_text_feats.ndim == 3:
#             ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, text_seq_length, -1)
#         elif ind_text_feats.ndim == 2:
#             ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, -1)
        
#         if ind_video_data.ndim == 3:
#             ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, video_seq_length, -1)
#         elif ind_video_data.ndim == 2:
#             ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, -1)
        
#         if ind_audio_data.ndim == 3:
#             ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, audio_seq_length, -1)
#         elif ind_audio_data.ndim == 2:
#             ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, -1)
            
#         mix_text = torch.cat((ind_text_feats, ood_text_feats), dim = 0)
#         mix_video = torch.cat((ind_video_data, ood_video_feats), dim = 0)
#         mix_audio = torch.cat((ind_audio_data, ood_audio_feats), dim = 0)

#         semi_label_ids = torch.cat((ind_label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
#         binary_label_ids = torch.cat((torch.tensor([1] * len(ind_text_feats)) , torch.tensor([0] * num_ood)), dim=0)
    
#         mix_data = {}
#         mix_data['text'] = mix_text.to(device)
#         mix_data['video'] = mix_video.to(device)
#         mix_data['audio'] = mix_audio.to(device)
#         mix_data['select_elems'] = select_elems

#         mix_labels = {
#             'ind': ind_label_ids.to(device),
#             'semi': semi_label_ids.to(device),
#             'binary': binary_label_ids.to(device)
#         }
       
#         return mix_data, mix_labels 

