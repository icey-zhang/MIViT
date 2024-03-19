# -*- coding:utf-8 -*-
# @Time       :2023/4/27 上午10.55
# @AUTHOR     :Jiaqing Zhang
# @FileName   :MMA.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure import decoder,classification,encoder_A_and_B, Mutual_info

class MMA(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.,fusion='trans'):
        super().__init__()


        self.decoder = decoder(l1, l2,patch_size, num_patches, encoder_embed_dim, decoder_embed_dim,
                 de_depth, de_heads, mlp_dim, dim_head, dropout, emb_dropout)
        self.encoder = encoder_A_and_B(l1, l2,patch_size, num_patches,num_classes, encoder_embed_dim,  en_depth, en_heads,dim_head, mlp_dim, dropout, emb_dropout,fusion)

        self.classifier = classification(encoder_embed_dim,num_classes)
        self.z1_z2_mutual = Mutual_info(64,64)
        self.z_z2_mutual = Mutual_info(64,64)
        self.z1_z_mutual = Mutual_info(64,64)
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda1
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda2


    def forward(self, img11, img21, img12, img22, img13, img23):

        x_vit, x_cnn,x1_out,x2_out,x1c_out,x_fuse1,x_fuse2,x_transfusion  = self.encoder(img11, img21, img12, img22, img13, img23, single_FLAG=0)
        con_loss = self.decoder(x_vit, img11, img21, img12, img22, img13, img23)
        x_cls,x_cls_cnn,x_cls_trans = self.classifier(x_vit, x_cnn)
        mutual_loss = (self.z1_z2_mutual(x_fuse1,x_fuse2) - self.xishu1 *self.z_z2_mutual(x_transfusion,x_fuse2) - self.xishu2*self.z1_z_mutual(x_fuse1,x_transfusion))[0]
        

        return x_cls,x_cls_cnn,x_cls_trans,x1_out,x2_out, con_loss,x1c_out,mutual_loss,x_fuse1,x_fuse2,x_transfusion

class MMA_test(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.,fusion='trans'):
        super().__init__()

        self.encoder = encoder_A_and_B(l1, l2,patch_size, num_patches,num_classes, encoder_embed_dim,  en_depth, en_heads,dim_head, mlp_dim, dropout, emb_dropout,fusion)
        self.classifier = classification(encoder_embed_dim,num_classes)


    def forward(self, img11, img21, img12, img22, img13, img23):

        x_vit, x_cnn,x1_out,x2_out,x1c_out,x_fuse1,x_fuse2,x_transfusion = self.encoder(img11, img21, img12, img22, img13, img23, single_FLAG=0)
        x_cls = self.classifier(x_vit, x_cnn)
        

        return x_cls