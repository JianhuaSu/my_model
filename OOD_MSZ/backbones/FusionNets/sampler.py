import torch
import numpy as np
from torch import nn


class DirSampler(nn.Module):

    def __init__(self, args):
        super(DirSampler, self).__init__()
        self.ood_label_id = args.ood_label_id
        self.args = args


    def __call__(self, ind_text_feats, ind_video_data, ind_audio_data, ind_label_ids, device=None):

            num_ood = int(len(ind_text_feats) * self.args.multiple_ood)

            ood_text_list, ood_video_list, ood_audio_list = [], [], []
            text_seq_length, video_seq_length, audio_seq_length = ind_text_feats.shape[1], ind_video_data.shape[1], ind_audio_data.shape[1]

            select_elems = []
            
            if ind_label_ids.size(0) >= 2:
                
                while len(ood_text_list) < num_ood:

                    if self.args.select_number_min == self.args.select_number_max:
                        select_number = self.args.select_number_min
                    else:    
                        select_number = np.random.randint(self.args.select_number_min, self.args.select_number_max + 1)

                    if ind_label_ids.size(0) <= select_number:
                        cdt = np.random.choice(ind_label_ids.size(0), select_number, replace=True)
                    else:
                        cdt = np.random.choice(ind_label_ids.size(0), select_number, replace=False)

                    if len(set(ind_label_ids[cdt].tolist())) >= 2:
                        
                        s = np.random.dirichlet(alpha=[self.args.alpha] * select_number)
                        
                        ood_text = sum(s[i] * ind_text_feats[cdt[i]] for i in range(select_number))
                        ood_video = sum(s[i] * ind_video_data[cdt[i]] for i in range(select_number))
                        ood_audio = sum(s[i] * ind_audio_data[cdt[i]] for i in range(select_number))

                        ood_text_list.append(ood_text)
                        ood_video_list.append(ood_video)
                        ood_audio_list.append(ood_audio)

                        # 记录选择的样本和混合比例
                        select_elems.append([cdt.tolist(), s.tolist()])
                    
                if ind_text_feats.ndim == 3:
                    ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, text_seq_length, -1)

                elif ind_text_feats.ndim == 2:
                    ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, -1)
                    
                if ind_video_data.ndim == 3:
                    ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, video_seq_length, -1)
                elif ind_video_data.ndim == 2:
                    ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, -1)
                
                if ind_audio_data.ndim == 3:
                    ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, audio_seq_length, -1)
                elif ind_audio_data.ndim == 2:
                    ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, -1)
                    
                mix_text = torch.cat((ind_text_feats, ood_text_feats), dim = 0)
                mix_video = torch.cat((ind_video_data, ood_video_feats), dim = 0)
                mix_audio = torch.cat((ind_audio_data, ood_audio_feats), dim = 0)

                semi_label_ids = torch.cat((ind_label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
                binary_label_ids = torch.cat((torch.tensor([1] * len(ind_text_feats)) , torch.tensor([0] * num_ood)), dim=0)

                mix_data = {}
                mix_data['text'] = mix_text.to(device)
                mix_data['video'] = mix_video.to(device)
                mix_data['audio'] = mix_audio.to(device)

            else:
                select_elems.append(1)
                semi_label_ids = ind_label_ids
                binary_label_ids = torch.tensor([1] * len(ind_text_feats))
                mix_data = {}
                mix_data['text'] = ind_text_feats.to(device)
                mix_data['video'] = ind_video_data.to(device)
                mix_data['audio'] = ind_audio_data.to(device)
                
            mix_labels = {
                'ind': ind_label_ids.to(device),
                'semi': semi_label_ids.to(device),
                'binary': binary_label_ids.to(device),
                'select_elems': select_elems
            }
        
            return mix_data, mix_labels 


class BetaSampler(nn.Module):

    def __init__(self, args):
        super(BetaSampler, self).__init__()
        self.ood_label_id = args.ood_label_id
        self.args = args

    def __call__(self, ind_text_feats, ind_video_data, ind_audio_data, ind_label_ids, device=None):
        
            num_ood = int(len(ind_text_feats) * self.args.multiple_ood)

            ood_text_list, ood_video_list, ood_audio_list = [], [], []
            text_seq_length, video_seq_length, audio_seq_length = ind_text_feats.shape[1], ind_video_data.shape[1], ind_audio_data.shape[1]

            select_elems = []

            while len(ood_text_list) < num_ood:
                
                cdt = np.random.choice(ind_label_ids.size(0), 2, replace=False)

                if len(set(ind_label_ids[cdt].tolist())) >= 2:

                    s = np.random.beta(self.args.alpha, self.args.alpha)  

                    ood_text = (s * ind_text_feats[cdt[0]] + (1 - s) * ind_text_feats[cdt[1]])
                    ood_video = (s * ind_video_data[cdt[0]] + (1 - s) * ind_video_data[cdt[1]])
                    ood_audio = (s * ind_audio_data[cdt[0]] + (1 - s) * ind_audio_data[cdt[1]])

                    # 选择具有最短序列长度的样本的掩码
                    ood_text_list.append(ood_text)
                    ood_video_list.append(ood_video)
                    ood_audio_list.append(ood_audio)

                    # 记录选择的样本和混合比例
                    select_elems.append([cdt[0], s, cdt[1]])

            if ind_text_feats.ndim == 3:
                ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, text_seq_length, -1)
            elif ind_text_feats.ndim == 2:
                ood_text_feats = torch.cat(ood_text_list, dim = 0).view(num_ood, -1)
                
            if ind_video_data.ndim == 3:
                ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, video_seq_length, -1)
            elif ind_video_data.ndim == 2:
                ood_video_feats = torch.cat(ood_video_list, dim = 0).view(num_ood, -1)
            
            if ind_audio_data.ndim == 3:
                ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, audio_seq_length, -1)
            elif ind_audio_data.ndim == 2:
                ood_audio_feats = torch.cat(ood_audio_list, dim = 0).view(num_ood, -1)
                
            mix_text = torch.cat((ind_text_feats, ood_text_feats), dim = 0)
            mix_video = torch.cat((ind_video_data, ood_video_feats), dim = 0)
            mix_audio = torch.cat((ind_audio_data, ood_audio_feats), dim = 0)

            semi_label_ids = torch.cat((ind_label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
            binary_label_ids = torch.cat((torch.tensor([1] * len(ind_text_feats)) , torch.tensor([0] * num_ood)), dim=0)

            mix_data = {}
            mix_data['text'] = mix_text.to(device)
            mix_data['video'] = mix_video.to(device)
            mix_data['audio'] = mix_audio.to(device)

            mix_labels = {
                'ind': ind_label_ids.to(device),
                'semi': semi_label_ids.to(device),
                'binary': binary_label_ids.to(device),
                'select_elems': select_elems
            }
        
            return mix_data, mix_labels 