import torch
import torch.nn as nn
import math
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from loss.cmd import CMD
from ..SubNets.AlignNets import AlignSubNet
import torch.nn.functional as F

from data.__init__ import benchmarks

__all__ = ['MSZ']

class MSZ(nn.Module):
    def __init__(self, args):
        super(MSZ, self).__init__()

        self.args = args

        # hidden_dims  dro   num_labels
        self.benchmarks = benchmarks[args.dataset]
        self.text_feat_dim, self.video_feat_dim, self.audio_feat_dim = self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        self.hidden_dims = args.hidden_dims
        self.att_audio_dro, self.att_video_dro, self.att_text_dro = args.att_dropouts
        self.spe_audio_dro, self.spe_video_dro, self.spe_text_dro = args.spe_dropouts
        self.output_dim = args.num_labels

        #loss
        self.cmd = CMD()
        self.criterion = nn.CrossEntropyLoss()

        self.att_ta = ATT(args)
        self.att_tv = ATT(args)
        self.att_av = ATT(args)
        self.att_at = ATT(args)
        self.att_vt = ATT(args)
        self.att_va = ATT(args)
        self.att_a = ATT(args)
        self.att_v = ATT(args)
        self.att_t = ATT(args)

        self.att_last = FAFWNet(3*768) 
        self.cf = CNNFusionBlock(3*768)

        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.hidden_dims)
        self.sigmoid = nn.Sigmoid()

        # transformer encoder
        self.text_bert = BERTEncoder.from_pretrained(args.text_backbone_path)
        self.audio_model = self.get_transformer_encoder(args, self.hidden_dims, args.encoder_layers)
        self.audio_liner = nn.Linear(self.audio_feat_dim, self.hidden_dims)
        self.video_model = self.get_transformer_encoder(args, self.hidden_dims, args.encoder_layers)
        self.video_liner = nn.Linear(self.video_feat_dim, self.hidden_dims)


        self.fusion_layer = nn.Linear(3*768, self.hidden_dims)
        self.fusion_layer_dropout =nn.Dropout(p=self.spe_text_dro)
        self.classifier = nn.Linear(self.hidden_dims, self.output_dim)

        self.con_linert = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.con_layer_dropout = nn.Dropout(p=self.spe_text_dro)

        encoder_layer = nn.TransformerEncoderLayer(d_model=3*args.hidden_dims, nhead=12)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.video_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.t_liner = nn.Linear(3*768, self.hidden_dims)
        self.a_liner = nn.Linear(3*768, self.hidden_dims)
        self.v_liner = nn.Linear(3*768, self.hidden_dims)

        if self.args.aligned_method:
            self.alignNet = AlignSubNet(args, args.aligned_method)


    def get_con(self, h):
        
        h = self.con_linert(h)
        h = self.activation(h)
        con = self.con_layer_dropout(h)

        return con  
    

    def get_transformer_encoder(self, args, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=args.nheads,
                                  layers=layers,
                                  attn_dropout=args.attn_dropout,
                                  relu_dropout=args.relu_dropout,
                                  res_dropout=args.res_dropout,
                                  embed_dropout=args.embed_dropout,
                                  attn_mask=args.attn_mask)

    def get_logits(self, h):
        
        h = self.fusion_layer(h)
        h = self.activation(h)
        h_last = self.fusion_layer_dropout(h)
  
        logits = self.classifier(h_last)

        return logits   

    def get_feats(self, text, audio, video):

        text = self.text_bert(text)
        
        if self.args.aligned_method:
            text, audio, video = self.alignNet(text, audio, video)

        text_mean = text[:, 0]

        audio = self.layernorm(self.activation(self.audio_model(self.audio_liner(audio.float().permute(1, 0, 2))).permute(1, 0, 2)))
        video = self.layernorm(self.activation(self.video_model(self.video_liner(video.float().permute(1, 0, 2))).permute(1, 0, 2)))


        audio_mean = audio[-1]
        video_mean = video[-1]

        output = {
            'text': text,
            'text_mean': text_mean,
            'audio': audio,
            'audio_mean': audio_mean,
            'video': video,
            'video_mean': video_mean,
        }

        return output
    

    def forward(self, text_feats, video_feats, audio_feats, label_ids):

        feats_output = self.get_feats(text_feats, audio_feats, video_feats)

        t_a_att_con = self.att_ta(feats_output['text'], feats_output['audio'])['sequence']
        t_v_att_con = self.att_tv(feats_output['text'], feats_output['video'])['sequence']
        t_att_con = self.att_t(feats_output['text'])['sequence']

        a_v_att_con = self.att_tv(feats_output['audio'], feats_output['video'])['sequence']
        a_t_att_con = self.att_ta(feats_output['audio'], feats_output['text'])['sequence']
        a_att_con = self.att_tv(feats_output['audio'])['sequence']

        v_a_att_con = self.att_ta(feats_output['video'], feats_output['audio'])['sequence']
        v_t_att_con = self.att_tv(feats_output['video'], feats_output['text'])['sequence']
        v_att_con = self.att_ta(feats_output['video'])['sequence']

        fusion_text = self.t_liner(self.text_encoder(torch.cat((t_v_att_con ,t_a_att_con, t_att_con), dim=2).permute(1, 0, 2)).permute(1, 0, 2))
        fusion_audio = self.a_liner(self.audio_encoder(torch.cat((a_v_att_con ,a_t_att_con, a_att_con), dim=2).permute(1, 0, 2)).permute(1, 0, 2))
        fusion_video = self.v_liner(self.video_encoder(torch.cat((v_a_att_con ,v_t_att_con, v_att_con), dim=2).permute(1, 0, 2)).permute(1, 0, 2))


        fusion_feats = torch.cat((fusion_text[:, -1, :], fusion_audio[:, -1, :], fusion_video[:, -1, :]), dim=1)
        # fusion_feats = torch.cat((fusion_text, fusion_audio, fusion_video), dim=2)

        fusion_feats = self.att_last(fusion_feats)
        # fusion_feats = self.cf(fusion_feats)

        logits = self.get_logits(fusion_feats)

        fusion_loss = self.criterion(logits, label_ids)

        outputs = {
            'logits':logits,
            'loss':fusion_loss
        }

        return outputs
    
    
    
class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim**-0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return  pos_emb
    
class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MeanPooling(nn.Module):
    """
    参考自: https://www.kaggle.com/code/quincyqiang/feedback-meanpoolingv2-inference
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_state.shape[0:-1])
            attention_mask = attention_mask.to(last_hidden_state.device)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AddNormalBlock(nn.Module):

    def __init__(self, dim):
        super(AddNormalBlock, self).__init__()

        # layernormal linear
        self.ln = nn.LayerNorm(dim)

    def forward(self, seq1, seq2):
        output_seq = seq1 + seq2
        output_seq = self.ln(output_seq)
        return output_seq


class FeedForward(nn.Module):

    def __init__(self, input_dim, inner_dim, outer_dim, drop_rate=0.2):
        super(FeedForward, self).__init__()

        # inner_linear
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.act = nn.GELU()

        # drop_layer
        self.drp_layer = nn.Dropout(p=drop_rate)

        # outer_linear
        self.fc2 = nn.Linear(inner_dim, outer_dim)

    def forward(self, input_seq):
        output_seq = self.act(self.fc1(input_seq))
        output_seq = self.drp_layer(output_seq)
        output_seq = self.fc2(output_seq)
        return output_seq


class EncoderLayer(nn.Module):

    def __init__(self,
                 dim,
                 heads=4,
                 att_drp=0.2,
                 feed_dim=768,
                 feed_drp=0.2,
                 batch_first=True):

        super(EncoderLayer, self).__init__()
        self.num_heads = heads
        self.layer = nn.ModuleDict({

            "att": nn.MultiheadAttention(dim, heads, att_drp, batch_first=batch_first),
            "add1": AddNormalBlock(dim),
            "feed": FeedForward(dim, feed_dim, dim, feed_drp),
            "add2": AddNormalBlock(dim),
        })

    def forward(self,
                input_seq,
                condition_seq=None):

        condition_seq = input_seq if condition_seq is None else condition_seq

        enc_en, enc_en_att = self.layer["att"](query=input_seq,
                                               key=condition_seq,
                                               value=condition_seq,
                                               key_padding_mask=None,
                                               need_weights=True,
                                               attn_mask=None,
                                            )
        # enc_en = self.layer["feed"](enc_en)

        enc_en_temp = self.layer["add1"](enc_en, input_seq)
        enc_en = self.layer["feed"](enc_en_temp)
        enc_en = self.layer["add2"](enc_en, enc_en_temp)

        return enc_en


class CAEncoder(nn.Module):

    def __init__(self, args):
        super(CAEncoder, self).__init__()
        self.encoder = nn.ModuleDict({})
        for i in range(args.num_hidden_layers):
            self.encoder["layer_{}".format(i)] = EncoderLayer(
                args.hidden_size, args.num_attention_heads, args.attention_probs_dropout_prob,
                args.intermediate_size, args.hidden_dropout_prob, args.batch_first)

    def forward(self,
                input_seq,
                condition_seq=None):
        x = input_seq
        for layer_name in self.encoder.keys():
            x = self.encoder[layer_name](x, condition_seq)
        return x

class CAEncoder(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,
                 intermediate_size,
                 num_hidden_layers,
                 batch_first=True):
        super().__init__()
        self.encoder = nn.ModuleDict({})
        for i in range(num_hidden_layers):
            self.encoder["layer_{}".format(i)] = EncoderLayer(
                hidden_size, num_attention_heads, attention_probs_dropout_prob,
                intermediate_size, hidden_dropout_prob, batch_first)

    def forward(self,
                input_seq,
                condition_seq=None):
        x = input_seq
        for layer_name in self.encoder.keys():
            x = self.encoder[layer_name](x, condition_seq)
        return x


class ATT(nn.Module):

    def __init__(self, args):
        super(ATT, self).__init__()

        if args.position_embedding_type == "absolute":
            self.pos_emb = AbsolutePositionalEmbedding(
                args.hidden_size, args.max_position_embeddings)
            
        elif args.position_embedding_type == "sin":
            self.pos_emb = SinusoidalPositionEmbedding(args.hidden_size)
        else:
            self.pos_emb = nn.Identity()

        self.encoder = CAEncoder(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            hidden_dropout_prob=args.hidden_dropout_prob,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_hidden_layers,
            batch_first=args.batch_first)
        self.pooler = MeanPooling()

    def forward(self, x, condition_seq=None):
        pos_emb = self.pos_emb(x)  # [B, T, D]
        x_input = x + pos_emb
        if condition_seq is not None:
            x_output = self.encoder(x_input,
                                    condition_seq=condition_seq)
        else:
            x_output = self.encoder(x_input)
        x_pooled = self.pooler(x_output)

        return {"sequence": x_output, "embedding": x_pooled}


class FAFWNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.relu(self.fc1(x))
        gates = self.sigmoid(self.fc2(gates))
        x = torch.mul(x, gates)
        return x


class CNNFusionBlock(nn.Module):
    def __init__(self, input_dim, in_channels=1, out_channels=768, kernel_heights=[1,3,5], dropout=0.5):
        super().__init__()

        embd_size = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
        nn.Linear(len(kernel_heights)*out_channels, embd_size),
        nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):

        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)

        return embd