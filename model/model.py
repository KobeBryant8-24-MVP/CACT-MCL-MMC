import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder
import math

__all__ = ['MMC']


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ['T']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.image_classfier = Classifier(args.img_dropout, args.img_out, args.post_dim, args.output_dim)
        if self.args.mmc not in ['V']:
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
            self.text_classfier = Classifier(args.text_dropout, args.text_out, args.post_dim, args.output_dim)
        #self.mm_classfier = Classifier(args.mm_dropout, args.text_out + args.img_out, args.post_dim, args.output_dim)
        self.mm_classfier = Classifier(args.mm_dropout, 400, args.post_dim, args.output_dim)
        
        #self.kl = nn.KLDivLoss(reduction="batchmean")        

        self.transformer1 = nn.Transformer(d_model=200, nhead=10,
                                           num_encoder_layers=10,
                                           num_decoder_layers=10,
                                           dim_feedforward=516,dropout=0.1)
     
        self.transformer2 = nn.Transformer(d_model=200, nhead=10,
                                           num_encoder_layers=10,
                                           num_decoder_layers=10,
                                           dim_feedforward=516, dropout=0.1)

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(768, 100)
        self.proj_text = nn.Linear(768, 100)

        self.proj_visual_bn = nn.BatchNorm1d(100)
        self.proj_text_bn = nn.BatchNorm1d(100)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(768, 100)
        self.layer_attn_text = nn.Linear(768, 100)

        #self.fc_as_self_attn = nn.Linear(800, 800)
        #self.self_attn_bn = nn.BatchNorm1d(800)

        # Classification layer
        #self.cls_layer = nn.Linear(800, 101)

    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        kl = nn.KLDivLoss(reduction="batchmean")
        feature_um = dict()
        output_um = dict()
        UMLoss = dict()

        text = self.text_encoder(text=text)  #(16,512,768)
       
        image = torch.squeeze(image, 1)
        image = self.image_encoder(pixel_values=image) #(16,197,768)
     
        output_text = self.text_classfier(text[:, 0, :])
        output_image = self.image_classfier(image[:, 0, :])
        
        e_i = text[:, 0, :]
        f_i = image[:, 0, :]

        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i)))  # N, dim_proj

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i))  # N, dim_proj

        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)

        joint_repr = torch.cat((masked_v_i, masked_e_i),
                               dim=1)  # N, 2*dim_proj

        maskedfi_2_32_100 = torch.stack([torch.cat((masked_v_i, f_i_tilde),
                               dim=1), torch.cat((masked_e_i, e_i_tilde),
                               dim=1)])
        maskedei_2_32_100 = torch.stack([torch.cat((masked_e_i, e_i_tilde),
                               dim=1), torch.cat((masked_v_i, f_i_tilde),dim=1)])

        posencoderfi_2_32_100 = maskedfi_2_32_100
        posencoderei_2_32_100 = maskedei_2_32_100

        # transformer1(src,tgt)的 参数 输入 要求是 三维tensor
        fi_ei_transformer1_fe = \
            self.transformer1(posencoderfi_2_32_100,
                              posencoderei_2_32_100)
   
        fi_ei_transformer2_ef= \
            self.transformer2(posencoderei_2_32_100,
                              posencoderfi_2_32_100)
        #mask_v_e_multiply = torch.multiply(masked_v_i, masked_e_i)
        #mask_v_e_add = masked_v_i + masked_e_i

        cat = torch.multiply(fi_ei_transformer1_fe[0]+fi_ei_transformer1_fe[1],
                         fi_ei_transformer2_ef[0]+fi_ei_transformer2_ef[1])
       
        joint_repr_add_trans = torch.cat((joint_repr, cat),
                                 dim=1)  # N, 4*dim_proj

        #fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
        #output_mm = self.mm_classfier(joint_repr)  #(16,101)
        output_mm = self.mm_classfier(joint_repr_add_trans)
        #output_mm =  self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint_repr)))))


        if infer:
            return output_mm

        MMLoss_m = torch.mean(criterion(output_mm, label))

        if self.args.mmc in ['NoMMC']:
            MMLoss_sum = MMLoss_m
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['SupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, label)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['UnSupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, None)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        MMLoss_text = torch.mean(criterion(output_text, label))
        MMLoss_image = torch.mean(criterion(output_image, label))
        #mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], output_text, output_image, label)
        #MMLoss_sum = MMLoss_text + MMLoss_image + MMLoss_m + 0.1 * mmcLoss

        #return MMLoss_sum, MMLoss_m, output_mm
        #return  MMLoss_m, output_mm 
        kl_loss = kl(F.log_softmax(output_image,dim=1),F.softmax(output_text,dim=1))+kl(F.log_softmax(output_image,dim=1),F.softmax(output_mm,dim=1))+kl(F.log_softmax(output_text,dim=1),F.softmax(output_mm,dim=1))

        return  MMLoss_m, MMLoss_text, MMLoss_image, kl_loss, output_mm


    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit

    def mmc_2(self, f0, f1, p0, p1, l):
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)

        if p0 is not None:
            p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)
            p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

        if l is None:
            return self.UnSupMMConLoss(f0, f1)
        elif p0 is None:
            return self.SupMMConLoss(f0, f1, l)
        else:
            return self.UniSMMConLoss(f0, f1, p0, p1, l)

    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

        return mean_log_pos

    def SupMMConLoss(self, feature_a, feature_b, labels, temperature=0.07):
        # compute the mask matrix
        labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
        mask = torch.eq(labels, labels.T).float()

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        return mean_log_pos.mean()

    def UnSupMMConLoss(self, feature_a, feature_b, temperature=0.07):

        # compute the mask matrix
        mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(self.args.device)

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        mean_log_pos = mean_log_pos.mean()

        return mean_log_pos

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, post_dim)
        self.post_layer_3 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_d = self.post_dropout(input_p1)
        input_p2 = F.relu(self.post_layer_2(input_d), inplace=False)
        output = self.post_layer_3(input_p2)
        return output


'''
class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ['T']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.image_classfier = Classifier(args.img_dropout, args.img_out, args.post_dim, args.output_dim)
        if self.args.mmc not in ['V']:
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
            self.text_classfier = Classifier(args.text_dropout, args.text_out, args.post_dim, args.output_dim)
        #self.mm_classfier = Classifier(args.mm_dropout, args.text_out + args.img_out, args.post_dim, args.output_dim)
        self.mm_classfier = Classifier(args.mm_dropout, 1736, args.post_dim, args.output_dim)

        self.dropout = Dropout()
        # Flatten image features to 1D array
        #self.flatten_vis = torch.nn.Flatten()

        self.pos_encoder = PositionalEncoding(200, 0.1)
        self.transformer1 = nn.Transformer(d_model=200, nhead=10,
                                           num_encoder_layers=10,
                                           num_decoder_layers=10,
                                           dim_feedforward=516, dropout=0.1)
       
        self.transformer2 = nn.Transformer(d_model=200, nhead=10,
                                           num_encoder_layers=10,
                                           num_decoder_layers=10,
                                           dim_feedforward=516, dropout=0.1)

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(768, 100)
        self.proj_text = nn.Linear(768, 100)

        self.proj_visual_bn = nn.BatchNorm1d(100)
        self.proj_text_bn = nn.BatchNorm1d(100)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(768, 100)
        self.layer_attn_text = nn.Linear(768, 100)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(4 * 100, 4 * 100)
        self.self_attn_bn = nn.BatchNorm1d(4 * 100)

        # Classification layer
        self.cls_layer = nn.Linear(4 * 100, 101)

    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        feature_um = dict()
        output_um = dict()
        UMLoss = dict()

        text = self.text_encoder(text=text)  #(16,512,768)
        #print("text:", text, text.shape)
        image = torch.squeeze(image, 1)
        image = self.image_encoder(pixel_values=image) #(16,197,768)
        #print("image:", image, image.shape)
        output_text = self.text_classfier(text[:, 0, :])
        output_image = self.image_classfier(image[:, 0, :])
        #print("text[:, 0, :]:", text[:, 0, :], text[:, 0, :].shape)
        #print("image[:, 0, :]:", image[:, 0, :], image[:, 0, :].shape)
        
        f_i = image[:, 0, :]  #(16,768)
        e_i = text[:, 0, :]   #(16,768)

        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i)))  # N, dim_proj

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i))  # N, dim_proj


        # The authors concatenated masked embeddings to get a joint representation
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)

        joint_repr = torch.cat((masked_v_i, masked_e_i),
                               dim=1)  # N, 2*dim_proj
       

        #fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1) #(16,1536)
        
        fusion = torch.cat([text[:, 0, :], image[:, 0, :],
                            joint_repr], dim=-1)  # (16,1536+200 = 1736)
                            
        #print("fusion:", fusion, fusion.shape)
        output_mm = self.mm_classfier(fusion)  #(16,101)
       
        #print("output_mm:", output_mm, output_mm.shape)
'''



'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder

__all__ = ['MMC']


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ['T']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.image_classfier = Classifier(args.img_dropout, args.img_out, args.post_dim, args.output_dim)
        if self.args.mmc not in ['V']:
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
            self.text_classfier = Classifier(args.text_dropout, args.text_out, args.post_dim, args.output_dim)
        self.mm_classfier = Classifier(args.mm_dropout, args.text_out + args.img_out, args.post_dim, args.output_dim)

    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        feature_um = dict()
        output_um = dict()
        UMLoss = dict()

        text = self.text_encoder(text=text)
        image = torch.squeeze(image, 1)
        image = self.image_encoder(pixel_values=image)
        output_text = self.text_classfier(text[:, 0, :])
        output_image = self.image_classfier(image[:, 0, :])

        fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
        output_mm = self.mm_classfier(fusion)

        if infer:
            return output_mm

        MMLoss_m = torch.mean(criterion(output_mm, label))

        if self.args.mmc in ['NoMMC']:
            MMLoss_sum = MMLoss_m
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['SupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, label)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['UnSupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, None)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        MMLoss_text = torch.mean(criterion(output_text, label))
        MMLoss_image = torch.mean(criterion(output_image, label))
        mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], output_text, output_image, label)
        MMLoss_sum = MMLoss_text + MMLoss_image + MMLoss_m + 0.1 * mmcLoss

        return MMLoss_sum, MMLoss_m, output_mm

    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit

    def mmc_2(self, f0, f1, p0, p1, l):
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)

        if p0 is not None:
            p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)
            p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

        if l is None:
            return self.UnSupMMConLoss(f0, f1)
        elif p0 is None:
            return self.SupMMConLoss(f0, f1, l)
        else:
            return self.UniSMMConLoss(f0, f1, p0, p1, l)

    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

        return mean_log_pos

    def SupMMConLoss(self, feature_a, feature_b, labels, temperature=0.07):
        # compute the mask matrix
        labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
        mask = torch.eq(labels, labels.T).float()

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        return mean_log_pos.mean()

    def UnSupMMConLoss(self, feature_a, feature_b, temperature=0.07):

        # compute the mask matrix
        mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(self.args.device)

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        mean_log_pos = mean_log_pos.mean()

        return mean_log_pos


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, post_dim)
        self.post_layer_3 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_d = self.post_dropout(input_p1)
        input_p2 = F.relu(self.post_layer_2(input_d), inplace=False)
        output = self.post_layer_3(input_p2)
        return output
'''





