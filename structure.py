# -*- coding:utf-8 -*-
# @Time       :2023/4/27 上午10.55
# @AUTHOR     :Jiaqing Zhang
# @FileName   :stucture.py
from cal_operator import up_conv_sig,conv_bn_relu,conv_bn_max_relu,comconv_bn_relu,comconv_bn_max_relu,comconv_bn_avg_relu,conv_bn_adpavg_relu
from cal_operator import conv_bn_relu_max
import torch.nn as nn
import torch
from cal_operator import Residual,PreNorm,FeedForward
from cal_operator import img2seq,seq2img
from cal_operator import ChannelAttention,SpatialAttention,TTOA
from cal_operator import block3d,block2d
from einops import rearrange, repeat
import torch.nn.functional as F
import kornia
class trans(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(trans, self).__init__()
        cl_channel = out_channels / 8
        cl_channel = int(cl_channel)
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.channle_module1 = ChannelAttention(out_channels)
        self.spatial_module1 = SpatialAttention()
        self.channle_module2 = ChannelAttention(out_channels)
        self.spatial_module2 = SpatialAttention()
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
    def forward(self, x1,x2):
        conv1 = self.con1(x1)
        conv2 = self.con2(x2)
        x1tx2 = self.channle_module1(conv1)
        # x1tx2_fu = self.spatial_module1(conv2 * x1tx2 + conv2)* x1tx2 + conv2
        x1tx2_fu = self.spatial_module1(x2 * x1tx2 + conv2)* x1tx2 + conv2 #小错误
        y2 = self.con3(x1tx2_fu)
        y1 = self.con4(conv1)
        x2tx1 = self.spatial_module2(y2)
        x2tx1_fu = self.channle_module2(conv1*x2tx1 + y1) * x2tx1 + y1
        return x2tx1_fu+y2

class trans_cs_sc(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(trans_cs_sc, self).__init__()
        cl_channel = out_channels / 8
        cl_channel = int(cl_channel)
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.channle_module1 = ChannelAttention(out_channels)
        self.spatial_module1 = SpatialAttention()
        self.channle_module2 = ChannelAttention(out_channels)
        self.spatial_module2 = SpatialAttention()
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
    def forward(self, x1,x2):
        conv1 = self.con1(x1)
        conv2 = self.con2(x2)
        c1 = self.channle_module1(conv1)
        x1tx2 = conv1 * c1 + conv1 #和自己做
        s1 = self.spatial_module1(x1tx2)
        x1tx2_fu = conv2 * s1 + conv2
        y2 = self.con3(x1tx2_fu)
        y1 = self.con4(conv1)
        s2 = self.spatial_module2(y2)
        x2tx1 = y2 * s2 + y2
        c2 = self.channle_module2(x2tx1)
        x2tx1_fu =  y1 * c2 + y1
        return x2tx1_fu+y2

class trans_s1_c1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(trans_s1_c1, self).__init__()
        cl_channel = out_channels / 8
        cl_channel = int(cl_channel)
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.channle_module1 = ChannelAttention(out_channels)
        self.spatial_module1 = SpatialAttention()
        # self.channle_module2 = ChannelAttention(out_channels)
        # self.spatial_module2 = SpatialAttention()
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
    def forward(self, x1,x2):
        conv1 = self.con1(x1)
        conv2 = self.con2(x2)
        x1tx2 = self.spatial_module1(conv1)
        x1tx2_fu = conv2 * x1tx2 + conv2#self.spatial_module1(conv2 * x1tx2 + conv2)* x1tx2 + conv2
        y2 = self.con3(x1tx2_fu)
        y1 = self.con4(conv1)
        x2tx1 = self.channle_module1(y2)
        x2tx1_fu = y1*x2tx1 + y1#self.channle_module2(y1*x2tx1 + y1) * x2tx1 + y1
        return x2tx1_fu+y2

class trans_c1_s1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(trans_c1_s1, self).__init__()
        cl_channel = out_channels / 8
        cl_channel = int(cl_channel)
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.channle_module1 = ChannelAttention(out_channels)
        self.spatial_module1 = SpatialAttention()
        # self.channle_module2 = ChannelAttention(out_channels)
        # self.spatial_module2 = SpatialAttention()
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
    def forward(self, x1,x2):
        conv1 = self.con1(x1)
        conv2 = self.con2(x2)
        x1tx2 = self.channle_module1(conv1)
        x1tx2_fu = conv2 * x1tx2 + conv2#self.spatial_module1(conv2 * x1tx2 + conv2)* x1tx2 + conv2
        y2 = self.con3(x1tx2_fu)
        y1 = self.con4(conv1)
        x2tx1 = self.spatial_module1(y2)
        x2tx1_fu = y1*x2tx1 + y1#self.channle_module2(y1*x2tx1 + y1) * x2tx1 + y1
        return x2tx1_fu+y2

class TTOA_trans(nn.Module):
    def __init__(self,low_channels,high_channels):
        super(TTOA_trans,self).__init__()
        self.trans1 = TTOA(low_channels,high_channels)
        self.trans2 = TTOA(high_channels,low_channels)
        self.conv = conv_bn_relu(low_channels,low_channels,3,1,1)
    def forward(self,x1,x2):
        x1_out = self.trans1(x1,x2)
        x2_out = self.trans2(x2,x1)
        out = self.conv(x1_out+x2_out)
        return out

class trans_TTOA_attention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(trans_TTOA_attention, self).__init__()
        cl_channel = out_channels / 8
        cl_channel = int(cl_channel)
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.attention_module1 = TTOA(out_channels,out_channels)
        self.attention_module2 = TTOA(out_channels,out_channels)
        # self.channle_module2 = ChannelAttention(out_channels)
        # self.spatial_module2 = SpatialAttention()
        self.con3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.001),  # note 默认可以修改
            nn.ReLU(),
        )
    def forward(self, x1,x2):
        conv1 = self.con1(x1)
        conv2 = self.con2(x2)
        x1tx2 = self.attention_module1(conv1,conv2)
        # x1tx2_fu = conv2 * x1tx2 + conv2#self.spatial_module1(conv2 * x1tx2 + conv2)* x1tx2 + conv2
        y2 = self.con3(x1tx2)
        y1 = self.con4(conv1)
        x2tx1 = self.attention_module2(y2,y1)
        # x2tx1_fu = y1 * x2tx1 + y1#self.channle_module2(y1*x2tx1 + y1) * x2tx1 + y1
        return x2tx1

class Multiconv(nn.Module):
    def __init__(self, l1, l2):
        super(Multiconv, self).__init__() 
        self.conv33_1 = conv_bn_relu(l1, l1,3,1,1)  
        self.conv55_1 = conv_bn_relu(l1, l1,5,1,2)  
        self.conv77_1 = conv_bn_relu(l1, l1,7,1,3) 
        self.conv33_2 = conv_bn_relu(l2, l2,3,1,1)  
        self.conv55_2 = conv_bn_relu(l2, l2,5,1,2)  
        self.conv77_2 = conv_bn_relu(l2, l2,7,1,3)   
    def forward(self, x_con1,x_con2):
        x31 = self.conv33_1(x_con1) 
        x51 = self.conv55_1(x_con1) 
        x71 = self.conv77_1(x_con1)  
        x32 = self.conv33_2(x_con2) 
        x52 = self.conv55_2(x_con2) 
        x72 = self.conv77_2(x_con2) 
        x_1 = x31+x51+x71
        x_2 = x32+x52+x72
        return x_1,x_2

class CNN_Decoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv3 = up_conv_sig(64, l1, 3, 1, 1,2)
        self.dconv4 = up_conv_sig(64, l2, 3, 1, 1,2)
        self.dconv5 = up_conv_sig(64, l1, 3, 1, 1,3)
        self.dconv6 = up_conv_sig(64, l2, 3, 1, 1,3)

    def forward(self, x_con1):
        x1 = self.dconv1(x_con1)
        x2 = self.dconv2(x_con1)

        x3 = self.dconv3(x_con1)
        x4 = self.dconv4(x_con1)

        x5 = self.dconv5(x_con1)
        x6 = self.dconv6(x_con1)
        return x1, x2, x3, x4, x5, x6

class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(conv_bn_adpavg_relu(64, 32, 1))
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(64, 32, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)

        return x_out

class Classifier_for_pyramid(nn.Module):
    def __init__(self, Classes,channel):
        super(Classifier_for_pyramid, self).__init__()

        self.conv1 = conv_bn_relu(channel*3, channel*2, 1,1,0)
        self.conv2 = conv_bn_adpavg_relu(channel*2, channel, 1)
        self.conv3 = nn.Sequential(nn.Conv2d(channel, Classes, 1))

    def forward(self, x1, x2, x3):
        x1_1 = self.conv1(x1)  # 64*192*8*8 -> 64*128*8*8
        x1_2 = self.conv2(x1_1)  # 64*128*8*8 -> 64*64*1*1
        x1_3 = self.conv3(x1_2)  # 64*64*1*1 -> 64*15*1*1
        x1_3 = x1_3.view(x1_3.size(0), -1)  # 64*15
        x1_out = F.softmax(x1_3, dim=1)

        return x1_out


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        
        return x


class decoder(nn.Module):
    def __init__(self, l1, l2,patch_size, num_patches, encoder_embed_dim, decoder_embed_dim,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.cnn_decoder = CNN_Decoder(l1, l2)
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, decoder_embed_dim))
        self.de_transformer = Transformer(decoder_embed_dim, de_depth, de_heads, dim_head, mlp_dim, dropout,
                                    num_patches)
        self.decoder_pred1 = nn.Linear(decoder_embed_dim, 64, bias=True)  # decoder to patch
        self.loss_fun2 = nn.MSELoss()
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        
    def forward(self,x, imgs11, imgs21, imgs12, imgs22, imgs13, imgs23):
 # embed tokens
        x = self.decoder_embedding(x)
        """ with or without decoder_pos_embed"""
        # add pos embed
        x += self.decoder_pos_embed

        x = self.de_transformer(x, mask=None)

        # predictor projection
        x_1 = self.decoder_pred1(x)

        # remove cls token
        x_con1 = x_1[:, 1:, :]

        x_con1 = torch.einsum('nld->ndl', x_con1)

        x_con1 = x_con1.reshape((x_con1.shape[0], x_con1.shape[1], self.patch_size, self.patch_size))

        x1, x2, x3, x4, x5, x6 = self.cnn_decoder(x_con1) #cnn_decoder

        # con_loss
        con_loss1 = 0.5 * (self.loss_fun2(x1, imgs11)) + 0.5 * (self.loss_fun2(x2, imgs21))
        con_loss2 = 0.5 * (self.loss_fun2(x3, imgs12)) + 0.5 * (self.loss_fun2(x4, imgs22))
        con_loss3 = 0.5 * (self.loss_fun2(x5, imgs13)) + 0.5 * (self.loss_fun2(x6, imgs23))
        con_loss = 1/3 * con_loss1 + 1/3 * con_loss2 + 1/3 * con_loss3

        return con_loss

class classification(nn.Module):
    def __init__(self,encoder_embed_dim,num_classes):
        super().__init__()
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )
    def forward(self, x, x_cnn):
        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        x_cls1 = self.mlp_head(x) #这是vit的分类器

        x_cnn = torch.einsum('ndl->nld', x_cnn)

        x_cls2 = self.cnn_classifier(seq2img(x_cnn)) #这是卷积的分类器

        x_cls = x_cls1 * self.coefficient1 + x_cls2 * self.coefficient2
        return x_cls,x_cls2 * self.coefficient2,x_cls1 * self.coefficient1 #是否需要融合

class classification1(nn.Module):
    def __init__(self,encoder_embed_dim,num_classes):
        super().__init__()
        # self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.cnn_classifier = CNN_Classifier(num_classes)
        # self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(encoder_embed_dim),
        #     nn.Linear(encoder_embed_dim, num_classes)
        # )
    def forward(self, x_cnn):
        # classification: using cls_token output
        # x = self.to_latent(x[:, 0])

        # MLP classification layer
        # x_cls1 = self.mlp_head(x)

        x_cnn = torch.einsum('ndl->nld', x_cnn)

        x_cls2 = self.cnn_classifier(seq2img(x_cnn))

        x_cls =  x_cls2 * self.coefficient2
        return x_cls

class cnn_encoder_A_and_B_cross(nn.Module):
    def __init__(self,l1,l2):
        super().__init__()
        self.conv1_1_cross = conv_bn_relu_max(l1, 64, 3, 1, 1,1)
        self.conv2_1_cross = conv_bn_relu_max(l2, 64, 3, 1, 1,1)
        self.conv1_2_cross = conv_bn_relu_max(l1, 64, 3, 1, 1,1)
        self.conv2_2_cross = conv_bn_relu_max(l2, 64, 3, 1, 1,1)
        self.conv1_3_cross = conv_bn_relu_max(l1, 64, 3, 1, 1,1)
        self.conv2_3_cross = conv_bn_relu_max(l2, 64, 3, 1, 1,1)
        # self.xishu_cross1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        # self.xishu_cross2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.trans1 = TTOA_trans(64,64)
        self.trans2 = TTOA_trans(64,64)
        self.trans3 = TTOA_trans(64,64)

    def forward(self, x11, x21, x12, x22, x13, x23):
        x1_1_cross = self.conv1_1_cross(x11)
        x2_1_cross = self.conv2_1_cross(x21)

        x1_2_cross = self.conv1_2_cross(x12)
        x2_2_cross = self.conv2_2_cross(x22)

        x1_3_cross = self.conv1_3_cross(x13)
        x2_3_cross = self.conv2_3_cross(x23)


        x_add_cross1 = self.trans1(x1_1_cross,x2_1_cross)
        x_add_cross2 = self.trans2(x1_2_cross,x2_2_cross)
        x_add_cross3 = self.trans3(x1_3_cross,x2_3_cross)
        return x_add_cross1,x_add_cross2,x_add_cross3



class cnn_encoder_A_and_B(nn.Module):
    def __init__(self, l1, l2):
        super(cnn_encoder_A_and_B, self).__init__()

        self.conv1 = conv_bn_relu(l1, 32, 3, 1, 1)
        self.conv2 = conv_bn_relu(l2, 32, 3, 1, 1)
        self.conv1_1 = conv_bn_relu_max(32, 64, 3, 1, 1,1)
        self.conv2_1 = conv_bn_relu_max(32, 64, 3, 1, 1,1)
        self.conv1_2 = conv_bn_relu_max(32, 64, 3, 1, 1,2)
        self.conv2_2 = conv_bn_relu_max(32, 64, 3, 1, 1,2)
        self.conv1_3 = conv_bn_relu_max(32, 64, 3, 1, 1,3)
        self.conv2_3 = conv_bn_relu_max(32, 64, 3, 1, 1,3)

        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.33]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.33]))  # 1 - lamda
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.33]))  # 1 - lamda


    def forward(self, x11, x21, x12, x22, x13, x23):

        x11 = self.conv1(x11) #64,32,8,8
        x21 = self.conv2(x21) #64,32,8,8
        x12 = self.conv1(x12) #64,32,16,16
        x22 = self.conv2(x22) #64,32,16,16
        x13 = self.conv1(x13) #64,32,24,24
        x23 = self.conv2(x23) #64,32,24,24

        x1_1 = self.conv1_1(x11) #64,64,8,8
        x1_2 = self.conv1_2(x12) #64,64,8,8
        x1_3 = self.conv1_3(x13) #64,64,8,8

        x_add1 = x1_1 * self.xishu1 + x1_2 * self.xishu2 + x1_3 * self.xishu3
        
        x2_1 = self.conv2_1(x21)
        x2_2 = self.conv2_2(x22)
        x2_3 = self.conv2_3(x23)
        x_add2 = x2_1 * self.xishu1 + x2_2 * self.xishu2 + x2_3 * self.xishu3
        # print('The start feature:','w0:',self.xishu1,'w1:',self.xishu2,'w2:',self.xishu3)

        return x_add1, x_add2


class encoder_A_and_B(nn.Module):
    def __init__(self,l1, l2, patch_size, num_patches,num_classes, encoder_embed_dim,  en_depth, en_heads,dim_head, mlp_dim, dropout=0., emb_dropout=0.,fusion='trans'):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, encoder_embed_dim))
        
        self.encoder_embedding1 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        # self.encoder_embedding2 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        # self.encoder_embedding3 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.en_transformer = Transformer(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                          num_patches)


        self.cnn_encoder = cnn_encoder_A_and_B(l1, l2)
        self.cnn_encoder_cross = cnn_encoder_A_and_B_cross(l1,l2)
        self.pyramid = pyramid(self.encoder_embed_dim)
        # self.cnn_encoder_single = CNN_Encoder_single(l1)
        self.classification1 = CNN_Classifier(num_classes)
        self.classification2 = CNN_Classifier(num_classes)
        self.classification_p = Classifier_for_pyramid(num_classes,64)
        # self.classificationc1 = CNN_Classifier(num_classes)
        # self.classificationc2 = CNN_Classifier(num_classes)
        # self.classificationc3 = CNN_Classifier(num_classes)

        if fusion == 'trans':
            self.trans = trans(64,64)
            print("the fusion method is trans")
        elif fusion == 'TTOA':
            self.trans = TTOA_trans(64,64)
            print("the fusion method is TTOA")
        elif fusion == 'trans_c1_s1':
            self.trans = trans_c1_s1(64,64)
            print("the fusion method is trans_c1_s1")
        elif fusion == 'trans_s1_c1':
            self.trans = trans_s1_c1(64,64)
            print("the fusion method is trans_s1_c1")
        elif fusion =='trans_TTOA_attention':
            self.trans = trans_TTOA_attention(64,64)
            print("the fusion method is trans_TTOA_attention")
        elif fusion =='trans_cs_sc':
            self.trans = trans_cs_sc(64,64)
            print("the fusion method is trans_cs_sc")
        # self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        # self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))
        
            
    def forward(self, x11, x21, x12, x22, x13, x23, single_FLAG):
        x_fuse1, x_fuse2, = self.cnn_encoder(x11, x21, x12, x22, x13, x23)  # x_fuse1:64*64*8*8, x_fuse2:64*64*4*4, x_fuse2:64*64*2*2
        x_add_cross1,x_add_cross2,x_add_cross3 = self.cnn_encoder_cross(x11, x21, x12, x22, x13, x23)
        x_pyramid1,x_pyramid2,x_pyramid3 = self.pyramid(x_add_cross1,x_add_cross2,x_add_cross3) #对不同尺度的两个模态信息进行融合
        x1_out = self.classification1(x_fuse1)
        x2_out = self.classification2(x_fuse2)
        x1c_out= self.classification_p(x_pyramid1,x_pyramid2,x_pyramid3)

        x_transfusion = self.trans(x_fuse1,x_fuse2) #对两个模态各自多尺度信息进行融合
        # x_cnn = self.xishu1*x_transfusion + self.xishu2*x_pyramid 
        x_cnn = x_transfusion#+x_pyramid1

        x_cnn = x_cnn.flatten(2)

        x_cnn = self.encoder_embedding1(x_cnn)

        x_cnn = torch.einsum('nld->ndl', x_cnn)

        b, n, _ = x_cnn.shape
        # add pos embed w/o cls token
        x = x_cnn #+ self.encoder_pos_embed[:, 1:, :]

        # append cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.encoder_pos_embed#[:, :1]
        x = self.dropout(x)

        x = self.en_transformer(x, mask=None)

        return x, x_cnn,x1_out,x2_out,x1c_out,x_fuse1,x_fuse2,x_transfusion
    
# to extract the multi-scale shared fatures
class Multiscale_extraction(nn.Module):
    def __init__(self,in_ch, out_ch):
        super().__init__()
        self.con1 = block3d(in_ch,out_ch,kernel=(3,5,5), stride=(1,1,1), padding=(1,1,1),padding1=(1,1,1),padding2=(1,0,0))
        # self.con1_1 = block3d(in_ch,out_ch,kernel=(3,5,5), stride=(1,1,1), padding=(1,1,1),padding1=(1,1,1),padding2=(1,0,0))
        self.con2 = block3d(in_ch,out_ch,kernel=(3,7,7), stride=(1,1,1), padding=(1,1,1),padding1=(1,0,0),padding2=(1,0,0))
        # self.con2_1 = block3d(in_ch,out_ch,kernel=(3,7,7), stride=(1,1,1), padding=(1,1,1),padding1=(1,0,0),padding2=(1,0,0))

    def forward(self,img13, img23):
        #shared features in the different scale
        img13 = img13.unsqueeze(1)
        img23 = img23.unsqueeze(1)
        img12 = self.con1(img13).squeeze()
        img22 = self.con1(img23).squeeze()
        img11 = self.con2(img13).squeeze()
        img21 = self.con2(img23).squeeze()
        img13 = img13.squeeze()
        img23 = img23.squeeze()
        return img11, img21, img12, img22, img13, img23

class Multiscale_extraction_2d(nn.Module):
    def __init__(self,l1, l2):
        super().__init__()
        self.con12 = block2d(l1,l1,kernel=(5,5), stride=(1,1), padding=(1,1))
        self.con22 = block2d(l2,l2,kernel=(5,5), stride=(1,1), padding=(1,1))
        self.con11 = block2d(l1,l1,kernel=(7,7), stride=(1,1), padding=(1,1),padding1=(0,0))
        self.con21 = block2d(l2,l2,kernel=(7,7), stride=(1,1), padding=(1,1),padding1=(0,0))
    def forward(self,img13, img23):
        #shared features in the different scale
        img12 = self.con12(img13)
        img22 = self.con22(img23)
        img11 = self.con11(img13)
        img21 = self.con21(img23)
        return img11, img21, img12, img22, img13, img23   

class pyramid(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv1 = conv_bn_relu_max(in_channel, in_channel, 3, 1, 1,1)
        self.up1 = nn.Upsample(scale_factor=2)  # add Upsample
        self.conv2c = conv_bn_relu_max(in_channel*2, in_channel, 3, 1, 1,1)
        self.conv2 = conv_bn_relu_max(in_channel, in_channel, 3, 1, 1,1)
        self.up2 = nn.Upsample(scale_factor=1.5)
        self.conv3c = conv_bn_relu_max(in_channel*2, in_channel, 3, 1, 1,1)
        self.conv3 = conv_bn_relu_max(in_channel, in_channel, 5, 1, 0,1)
        self.conv3_ = conv_bn_relu_max(in_channel, in_channel, 5, 1, 0,1)
        self.conv4c = conv_bn_relu_max(in_channel*2, in_channel*2, 3, 1, 1,1)
        self.conv4 = conv_bn_relu_max(in_channel*2, in_channel*2, 3, 1, 1,2)
        self.conv5 = conv_bn_relu_max(in_channel*3, in_channel*3, 3, 1, 1,1)
        
    def forward(self,cross1,cross2,cross3):
        x1 = self.conv1(cross1) 
        feature1 = self.up1(x1)
        feature_cat1 = torch.cat([feature1,cross2],dim=1)
        x2c = self.conv2c(feature_cat1)
        x2 = self.conv2(x2c)
        feature2 = self.up2(x2)
        feature_cat2 = torch.cat([feature2,cross3],dim=1)
        x3c = self.conv3c(feature_cat2)
        x3 = self.conv3_(self.conv3(x3c))
        

        fature_cat3 = torch.cat([x3,x2],dim=1)
        x4c = self.conv4c(fature_cat3)
        x4 = self.conv4(x4c)
        fature_cat4 = torch.cat([x4,x1],dim=1)
        x5 = self.conv5(fature_cat4)
        
        return x5,x4c,x3c

from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='mean')
class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.channel = channels

        self.fc1_rgb3 = nn.Linear(channels*1*8*8, latent_size)
        self.fc2_rgb3 = nn.Linear(channels*1*8*8, latent_size)
        self.fc1_depth3 = nn.Linear(channels*1*8*8, latent_size)
        self.fc2_depth3 = nn.Linear(channels*1*8*8, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        rgb_feat = rgb_feat.view(-1, self.channel * 1*8*8 )
        depth_feat = depth_feat.view(-1, self.channel * 1*8*8)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld

        return latent_loss


class Mutual_info(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)

        self.channel = channels

        self.fc1_rgb3 = nn.Linear(channels*1*8*8, latent_size)
        self.fc1_depth3 = nn.Linear(channels*1*8*8, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def cal_loss(self,f_s, f_t,reduction='mean'):
        p_s = F.log_softmax(f_s, dim=1)
        p_t = F.softmax(f_t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction=reduction) / f_t.shape[0]
        return loss
    def cal_loss2(self,z_rgb,z_depth,reduction='mean'):
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        # CE = torch.nn.BCELoss(reduction=reduction)
        CE = torch.nn.CrossEntropyLoss(reduction=reduction)
        loss = CE(z_depth_norm, z_rgb_norm.detach())
        return loss
    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.leakyrelu(self.layer1(rgb_feat))
        depth_feat = self.leakyrelu(self.layer2(depth_feat))
        rgb_feat = rgb_feat.view(-1, self.channel * 1*8*8 )
        depth_feat = depth_feat.view(-1, self.channel * 1*8*8)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        z_rgb = mu_rgb
        dist_rgb =mu_rgb
        z_depth =mu_depth
        dist_depth =mu_depth
        bi_di_kld = self.cal_loss(dist_rgb, dist_depth)+self.cal_loss(dist_depth, dist_rgb)
        bi_ce = self.cal_loss2(z_rgb,z_depth)+self.cal_loss2(z_depth,z_rgb)
        latent_loss = bi_ce-bi_di_kld

        return latent_loss
