import torch
import torch.nn as nn
import math
from ..SubNets.FeatureNets import BERTEncoder
from transformers import BertForSequenceClassification
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from loss.angular import AngularPenaltySMLoss
from loss.cmd import CMD
from ..SubNets.AlignNets import AlignSubNet

from data.__init__ import benchmarks

__all__ = ['MSZ']

class MSZ(nn.Module):
    def __init__(self, args):
        super(MSZ, self).__init__()

        self.args = args
        
        # transformer params
        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask

        # hidden_dims  dro   num_labels
        self.benchmarks = benchmarks[args.dataset]
        self.text_feat_dim, self.video_feat_dim, self.audio_feat_dim = self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        self.hidden_dims = args.hidden_dims
        self.att_audio_dro, self.att_video_dro, self.att_text_dro = args.att_dropouts
        self.spe_audio_dro, self.spe_video_dro, self.spe_text_dro = args.spe_dropouts
        self.output_dim = args.num_labels

        #loss
        self.cosface = AngularPenaltySMLoss(args, loss_type='cosface')
        self.cmd = CMD()
        self.criterion = nn.CrossEntropyLoss()
        self.att = ATT(args)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.hidden_dims)
        self.sigmoid = nn.Sigmoid()

        # transformer encoder
        self.text_bert = BERTEncoder.from_pretrained(args.text_backbone_path)
        self.text_model = self.get_transformer_encoder(self.hidden_dims, args.encoder_layers)
        self.text_liner = nn.Linear(self.text_feat_dim, self.hidden_dims)
        self.audio_model = self.get_transformer_encoder(self.hidden_dims, args.encoder_layers)
        self.audio_liner = nn.Linear(self.audio_feat_dim, self.hidden_dims)
        self.video_model = self.get_transformer_encoder(self.hidden_dims, args.encoder_layers)
        self.video_liner = nn.Linear(self.video_feat_dim, self.hidden_dims)

        # att liner
        self.t_a_att_dropout = nn.Dropout(p=self.att_video_dro)
        self.t_a_att_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.a_t_att_dropout = nn.Dropout(p=self.att_video_dro)
        self.a_t_att_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.t_v_att_dropout = nn.Dropout(p=self.att_video_dro)
        self.t_v_att_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.v_t_att_dropout = nn.Dropout(p=self.att_video_dro)
        self.v_t_att_liner = nn.Linear(self.hidden_dims, self.hidden_dims)

        # special liner
        self.t_a_spe_dropout = nn.Dropout(p=self.spe_text_dro)
        self.t_a_spe_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.t_v_spe_dropout = nn.Dropout(p=self.spe_text_dro)
        self.t_v_spe_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.v_spe_dropout = nn.Dropout(p=self.spe_video_dro)
        self.v_spe_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.a_spe_dropout = nn.Dropout(p=self.spe_video_dro)
        self.a_spe_liner = nn.Linear(self.hidden_dims, self.hidden_dims)

        # fuison liner
        self.f1_dropout = nn.Dropout(p=self.spe_text_dro)
        self.f1_liner = nn.Linear(3 * self.hidden_dims, self.hidden_dims)
        self.f2_dropout = nn.Dropout(p=self.spe_text_dro)
        self.f2_liner = nn.Linear(self.hidden_dims, self.hidden_dims)
        
        # common liner
        self.con_linert = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.con_layer_t_dropout = nn.Dropout(0.1)
        self.con_linera = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.con_layer_a_dropout = nn.Dropout(0.1)        
        self.con_linerv = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.con_layer_v_dropout = nn.Dropout(0.1)

        self.fusion_layer_1 = nn.Linear(768, self.hidden_dims)
        self.fusion_layer_1_dropout =nn.Dropout(p=self.spe_text_dro)
        self.classifier = nn.Linear(self.hidden_dims, self.output_dim)


        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_dims, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if self.args.aligned_method:
            self.alignNet = AlignSubNet(args, args.aligned_method)


    def get_t(self, h):
        
        h = self.con_linert(h)
        h = self.activation(h)
        con = self.con_layer_t_dropout(h)

        return con  

    def get_a(self, h):
        
        h = self.con_linera(h)
        h = self.activation(h)
        con = self.con_layer_a_dropout(h)

        return con     
     
    def get_v(self, h):
        
        h = self.con_linerv(h)
        h = self.activation(h)
        con = self.con_layer_v_dropout(h)

        return con  
    
    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_logits(self, h):
        
        h = self.fusion_layer_1(h)
        h = self.activation(h)
        h_last = self.fusion_layer_1_dropout(h)
  
        logits = self.classifier(h_last)

        return logits   
    
    def get_feats(self, feats, m_type=None):

        # feats [16, 30, 768]
        if m_type=='text':
            feats = self.text_bert(feats)
            feats = self.text_liner(feats)
            feats = feats.permute(1, 0, 2)
            feats = self.text_model(feats)
        elif m_type=='video':
            feats = feats.float()
            feats = self.video_liner(feats)
            feats = feats.permute(1, 0, 2)
            feats = self.video_model(feats)
        elif m_type=='audio':
            feats = feats.float()
            feats = self.audio_liner(feats)
            feats = feats.permute(1, 0, 2)
            feats = self.audio_model(feats)

        feats = feats.permute(1, 0, 2)
        feats = self.layernorm(self.activation(feats))
        feats_mean = feats.mean(dim = 1)

        return feats, feats_mean
    
    def forward(self, text_feats, video_feats, audio_feats, label_ids):

        text, text_mean = self.get_feats(text_feats, m_type='text')
        video, video_mean = self.get_feats(video_feats, m_type='video')
        audio, audio_mean = self.get_feats(audio_feats, m_type='audio')
        
        if self.args.aligned_method:
            text, audio, video = self.alignNet(text, audio, video)
            text_mean = text.mean(dim = 1)
            audio_mean = audio.mean(dim = 1)
            video_mean = video.mean(dim = 1)

        # attention [16,768]
        t_a_att = self.layernorm(self.activation(self.t_a_att_liner(self.t_a_att_dropout(self.att(text, audio)))))
        a_t_att = self.layernorm(self.activation(self.a_t_att_liner(self.a_t_att_dropout(self.att(audio, text)))))
        t_v_att = self.layernorm(self.activation(self.t_v_att_liner(self.t_v_att_dropout(self.att(text, video)))))
        v_t_att = self.layernorm(self.activation(self.v_t_att_liner(self.v_t_att_dropout(self.att(video, text)))))

        text_con = self.get_t(text_mean)
        video_con = self.get_v(video_mean)
        audio_con = self.get_a(audio_mean)

        t_a_att_con = self.get_a(t_a_att)
        a_t_att_con = self.get_t(a_t_att)
        t_v_att_con = self.get_v(t_v_att)
        v_t_att_con = self.get_t(v_t_att)

        tv_c_loss = self.cmd(a_t_att_con, v_t_att_con, 5)
        vt_c_loss = self.cmd(video_con, t_v_att_con, 5)
        at_c_loss = self.cmd(audio_con, t_a_att_con, 5)

        # text_con = self.get_t(text_mean)
        # video_con = self.get_v(video_mean)
        # audio_con = self.get_a(audio_mean)
        
        # t_a_att_con = self.get_v(t_a_att)
        # a_t_att_con = self.get_a(a_t_att)
        # t_v_att_con = self.get_t(t_v_att)
        # v_t_att_con = self.get_v(v_t_att)

        # # ta_c_loss = self.cmd(text_con, a_t_att_con, 2)
        # tv_c_loss = self.cmd(t_v_att_con, t_a_att_con, 2)
        # vt_c_loss = self.cmd(video_con, v_t_att_con, 2)
        # at_c_loss = self.cmd(audio_con, a_t_att_con, 2)

        # print(ta_c_loss)
        # print(tv_c_loss)
        print(vt_c_loss)
        print(at_c_loss)

        # t_att = self.t_a_att_liner(self.sigmoid(self.t_a_att_dropout(self.att(text))))
        # text = self.att(text)
        # # special t a v   [16,768]
        # t_v_spe_feat = self.t_a_spe_liner(self.t_a_spe_dropout(text_mean - t_v_att - v_t_att))
        # t_a_spe_feat = self.t_v_spe_liner(self.t_v_spe_dropout(text_mean - t_a_att - a_t_att))
        # a_spe_feat = self.v_spe_liner(self.v_spe_dropout(audio_mean - t_a_att - a_t_att))
        # v_spe_feat = self.a_spe_liner(self.a_spe_dropout(video_mean - t_v_att - v_t_att))
        # fusion_feats = torch.cat((text_mean, audio_mean, video_mean), dim = 1)
        # fusion
        # fusion_feats = torch.cat((self.common_liner(self.activation(t_a_att)), self.common_liner(self.activation(a_t_att)), self.common_liner(self.activation(t_v_att)), self.common_liner(self.activation(v_t_att)), self.common_liner(t_v_spe_feat), self.common_liner(t_a_spe_feat), self.common_liner(a_spe_feat), self.common_liner(v_spe_feat)), dim = 1)
        # fusion_feats = torch.cat((self.get_con(text_mean), self.get_con(video_mean), self.get_con(audio_mean)), dim = 1)
        # fusion_feats = self.activation(self.f2_liner(self.f1_dropout(self.f1_liner(fusion_feats))))
        # fusion_feats = self.f1_dropout(self.f1_liner(fusion_feats))
        # fusion_feats = self.get_con(text_mean) + self.get_con(video_mean) + self.get_con(audio_mean)
        # fusion_feats = self.activation(fusion_feats)

        fusion_feats = self.activation((t_a_att_con + a_t_att_con + text_con)/3 + (t_v_att_con + v_t_att_con + text_con)/3)

        fusion_feats = self.transformer_encoder(fusion_feats)
   
        # # loss
        # s_feature = torch.cat([t_v_spe_feat,t_a_spe_feat,a_spe_feat,v_spe_feat],dim=0)
        # s_label =  torch.cat([torch.ones_like(torch.mean(t_v_spe_feat,dim=1))*i for i in range(4)],dim=0).long().to(self.args.device)
        # specific_loss = self.cosface(s_feature, s_label)

        # ta_c_loss = self.cmd(t_a_att, a_t_att, 2)
        # tv_c_loss = self.cmd(t_v_att, v_t_att, 2)

        logits = self.get_logits(fusion_feats)    
        fusion_loss = self.criterion(logits, label_ids)

        # loss = self.args.tv_c_loss * tv_c_loss + self.args.vt_c_loss * vt_c_loss + self.args.at_c_loss*at_c_loss + fusion_loss

        loss = self.args.vt_c_loss * vt_c_loss + self.args.at_c_loss*at_c_loss + fusion_loss


        outputs = {
            # 'fusion_feats': fusion_feats,
            'logits':logits,
            'loss':loss
        #     'loss': loss,
        #     'c_loss':ta_c_loss + tv_c_loss,
        #     's_loss':specific_loss,
        #     'f_loss':fusion_loss
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
        # TODO 需要考虑加position embedding, 如果要用的话
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
        self.add_cross_attention = args.add_cross_attention

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
        if self.add_cross_attention:
            x_output = self.encoder(x_input,
                                    condition_seq=condition_seq)
        else:
            x_output = self.encoder(x_input)
        x_pooled = self.pooler(x_output)
        # return {"sequence": x_output, "embedding": x_pooled}
        return x_pooled