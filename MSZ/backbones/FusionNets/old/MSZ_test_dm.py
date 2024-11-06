import torch
import torch.nn as nn
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.AlignNets import AlignSubNet

from data.__init__ import benchmarks

__all__ = ['MSZ']

class MSZ(nn.Module):
    def __init__(self, args):
        super(MSZ, self).__init__()

        self.args = args

        # hidden_dims  dro   num_labels
        self.benchmarks = benchmarks[args.dataset]
        self.text_feat_dim, self.video_feat_dim, self.audio_feat_dim = self.benchmarks['feat_dims']['text'], \
                                                self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.output_dim = args.num_labels

        #loss
        self.criterion = nn.CrossEntropyLoss()

        self.att_ta = ATT(args, args.m_num_attention_heads, args.m_attention_probs_dropout_prob, args.m_hidden_dropout_prob,args.m_num_hidden_layers, args.m_max_position_embeddings)
        self.att_tv = ATT(args, args.m_num_attention_heads, args.m_attention_probs_dropout_prob, args.m_hidden_dropout_prob,args.m_num_hidden_layers, args.m_max_position_embeddings)
        self.att_a = ATT(args, args.d_num_attention_heads, args.d_attention_probs_dropout_prob, args.d_hidden_dropout_prob,args.d_num_hidden_layers, args.d_max_position_embeddings)
        self.att_v = ATT(args, args.d_num_attention_heads, args.d_attention_probs_dropout_prob, args.d_hidden_dropout_prob,args.d_num_hidden_layers, args.d_max_position_embeddings)
        self.att_t = ATT(args, args.d_num_attention_heads, args.d_attention_probs_dropout_prob, args.d_hidden_dropout_prob,args.d_num_hidden_layers, args.d_max_position_embeddings)

        self.att_last = FATT(3*self.hidden_dims) 

        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.hidden_dims)
        # self.tag_layernorm = nn.LayerNorm(2 * self.hidden_dims)

        # transformer encoder
        self.text_bert = BERTEncoder.from_pretrained(args.text_backbone_path)
        self.audio_liner = nn.Linear(self.audio_feat_dim, self.hidden_dims)
        self.video_liner = nn.Linear(self.video_feat_dim, self.hidden_dims)

        self.fusion_layer = nn.Linear(3 * self.hidden_dims, self.hidden_dims)
        self.fusion_layer_dropout =nn.Dropout(p=self.dropout)
        self.classifier = nn.Linear(self.hidden_dims, self.output_dim)

        self.ta_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)
        self.tv_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)
        self.t_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)

        self.at_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)
        self.aa_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)
        self.av_liner = nn.Linear(2 * self.hidden_dims, self.hidden_dims)

        if self.args.aligned_method:
            self.alignNet = AlignSubNet(args, args.aligned_method)


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

        text = self.att_t(text)
        audio = self.att_a(self.audio_liner(audio.float()))
        video = self.att_v(self.video_liner(video.float()))

        text, video, audio = self.add_modality_info(text['sequence'], video['sequence'], audio['sequence'])

        output = {
            'text': self.layernorm(self.activation(self.at_liner(text))),
            'audio': self.layernorm(self.activation(self.aa_liner(audio))),
            'video': self.layernorm(self.activation(self.av_liner(video)))
        }
        return output
    

    def add_modality_info(self, text_data, video_data, audio_data):
        batch_size = text_data.size(0)
        seq_size = text_data.size(1)
        hidden_size = text_data.size(2)

        text_tag = torch.ones(batch_size, seq_size, hidden_size, device=self.args.device)
        video_tag = torch.ones(batch_size, seq_size, hidden_size, device=self.args.device) * 2
        audio_tag = torch.ones(batch_size, seq_size, hidden_size, device=self.args.device) * 3

        text_concat = torch.cat((text_data, text_tag), dim=2)
        video_concat = torch.cat((video_data, video_tag), dim=2)
        audio_concat = torch.cat((audio_data, audio_tag), dim=2)

        return text_concat, video_concat, audio_concat


    def forward(self, text_feats, video_feats, audio_feats, label_ids):

        feats_output = self.get_feats(text_feats, audio_feats, video_feats)

        text, video, audio = self.add_modality_info(feats_output['text'],feats_output['video'], feats_output['audio'])

        fusion_ta = torch.cat((text, audio), dim=1)
        fusion_tv = torch.cat((text, video), dim=1)
        fusion_ta = self.layernorm(self.activation(self.ta_liner(fusion_ta)))
        fusion_tv = self.layernorm(self.activation(self.tv_liner(fusion_tv)))

        t_a_att_con = self.att_ta(fusion_tv, self.t_liner(text))['embedding']
        t_v_att_con = self.att_tv(fusion_ta, self.t_liner(text))['embedding']
        t_con = self.att_t(self.t_liner(text))['embedding']

        fusion_feats = torch.cat((t_a_att_con, t_v_att_con, t_con), dim=1)
        fusion_feats = self.att_last(fusion_feats)
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
        assert seq_len <= self.max_seq_len

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return  pos_emb
    

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
        self.lay = nn.LayerNorm(dim)

    def forward(self, seq1, seq2):
        output = seq1 + seq2
        output = self.lay(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, input_dim, inner_dim, outer_dim, drop_rate=0.2):
        super(FeedForward, self).__init__()

        self.fl1 = nn.Linear(input_dim, inner_dim)
        self.act = nn.GELU()
        self.drp_layer = nn.Dropout(p=drop_rate)
        self.fl2 = nn.Linear(inner_dim, outer_dim)

    def forward(self, input):
        output = self.act(self.fl1(input))
        output = self.drp_layer(output)
        output = self.fl2(output)
        return output


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

            "m_att": nn.MultiheadAttention(dim, heads, att_drp, batch_first=batch_first),
            "norm1": AddNormalBlock(dim),
            "feed": FeedForward(dim, feed_dim, dim, feed_drp),
            "norm2": AddNormalBlock(dim),
        })

    def forward(self,
                input_seq,
                condition_seq=None):

        condition_seq = input_seq if condition_seq is None else condition_seq

        att, _ = self.layer["m_att"](query=input_seq,
                                               key=condition_seq,
                                               value=condition_seq,
                                               key_padding_mask=None,
                                               need_weights=True,
                                               attn_mask=None,
                                            )

        n_att = self.layer["norm1"](att, input_seq)
        att = self.layer["feed"](n_att)
        att = self.layer["norm2"](att, n_att)

        return att


class Encoder(nn.Module):

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

    def __init__(self, args, 
                 num_attention_heads, 
                 attention_probs_dropout_prob, 
                 hidden_dropout_prob,
                 num_hidden_layers, 
                 max_position_embeddings):
        super(ATT, self).__init__()

        if args.position_embedding_type == "absolute":
            self.pos_emb = AbsolutePositionalEmbedding(
                args.hidden_dims, max_position_embeddings)
        else:
            self.pos_emb = nn.Identity()

        self.encoder = Encoder(
            hidden_size=args.hidden_dims,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=num_hidden_layers,
            batch_first=args.batch_first)
        self.pooler = MeanPooling()

    def forward(self, x, condition_seq=None):
        pos_emb = self.pos_emb(x) 
        x_input = x + pos_emb
        if condition_seq is not None:
            x_output = self.encoder(x_input,
                                    condition_seq=condition_seq)
        else:
            x_output = self.encoder(x_input)
        x_pooled = self.pooler(x_output)

        return {"sequence": x_output, "embedding": x_pooled}

class FATT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fa1 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.relu = nn.ReLU()
        self.fa2 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.relu(self.fa1(x))
        weight = self.sigmoid(self.fa2(weight))
        x = torch.mul(x, weight)
        return x