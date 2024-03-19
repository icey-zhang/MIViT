# -*- coding:utf-8 -*-
# @Time       :2023/4/27 上午10.55
# @AUTHOR     :Jiaqing Zhang
# @FileName   :demo.py
import torch.nn as nn
import torch.nn.functional as F
import torch
def gn_relu(in_channel, num_group):
    return nn.Sequential(
        nn.GroupNorm(num_group, in_channel),
        nn.ReLU(inplace=True),
    )  

def up_conv_sig(in_channel, out_channel,kernel_size=3, stride=1, padding=1,scale_factor=2):
    return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),  # add Upsample
                nn.Conv2d(in_channel,out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Sigmoid(),
    )
def conv_bn_relu(in_channel, out_channel,kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),  # todo: paddint
        nn.BatchNorm2d(out_channel, momentum=0.9, eps=0.001),  # note 默认可以修改
        nn.ReLU()
    )

def conv_bn_max_relu(in_channel, out_channel,max_kernel):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel, momentum=0.9, eps=0.001),
            nn.MaxPool2d(kernel_size=max_kernel, stride=2, padding=1),
            nn.ReLU()
        )

def conv_bn_relu_max(in_channel, out_channel,kernel_size=3, stride=1, padding=1,max_kernel=2):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(max_kernel),
        )

def conv_bn_adpavg_relu(in_channel, out_channel,kernel_size=1):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),   
        )

def comconv_bn_relu(fusion_conv1,channels):
    return nn.Sequential(
                fusion_conv1,
                nn.BatchNorm2d(channels, momentum=0.9, eps=0.001),
                nn.ReLU()
            )
def comconv_bn_max_relu(common_con,channels):
        return nn.Sequential(
            common_con,
            nn.BatchNorm2d(channels, momentum=0.9, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )


def comconv_bn_avg_relu(common_con,channels):
        return nn.Sequential(
            common_con,
            nn.BatchNorm2d(channels, momentum=0.9, eps=0.001),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h*w))
    return x

def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class Residual_block(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1): # use_1x1conv表示是否使用1*1的卷积层
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1) # 第二个卷积层stride默认等于1
        if use_1x1conv: # 如果使用1*1的卷积层
        # 相当于对输入x处理形状，使其匹配残差块的输出（为了能相加）
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class TTOA(nn.Module):
    def __init__(self,low_channels,high_channels,c_kernel=3,r_kernel=3,use_att=False,use_process=True):
        '''
                  :param low_channels: low_level feature channels
                  :param high_channels: high_level feature channels
                  :param c_kernel: colum dcn kernels kx1 just use k
                  :param r_kernel: row dcn kernels 1xk just use k
                  :param use_att: bools
                  :param use_process: bools
                  '''
        super(TTOA, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        if self.l_c == self.h_c:
            print('Channel checked!')
        else:
            raise ValueError('Low and Hih channels need to be the same!')
        self.dcn_row = nn.Conv2d(self.l_c,self.h_c,kernel_size=(1,self.r_k),stride=1,padding=(0,self.r_k//2))
        self.dcn_colum = nn.Conv2d(self.l_c,self.h_c,kernel_size=(self.c_k,1),stride=1,padding=(self.c_k//2,0))
        self.sigmoid = nn.Sigmoid()
        if self.att == True:
            self.csa = self.non_local_att(self.l_c,self.h_c,1,1,0)
        else:
            self.csa = None
        if use_process == True:
            self.preprocess = nn.Sequential(nn.Conv2d(self.l_c,self.h_c//2,1,1,0),nn.Conv2d(self.h_c//2,self.l_c,1,1,0))
        else:
            self.preprocess = None
    def forward(self,a_low,a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn_colum(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        a_low_r = self.dcn_row(a_low)
        a_low_rw = self.sigmoid(a_low_r)
        a_low_rw = a_low_rw * a_high
        a_row = a_low + a_low_rw

        if self.csa is not None:
            a_TTOA = self.csa(a_row + a_colum)
        else:
            a_TTOA = a_row + a_colum
        return a_TTOA

class block3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(4,3,3), stride=(2,1,1), padding=(1,1,1),padding1=(1,1,1),padding2=(1,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel,stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=kernel,stride=stride,padding=padding1)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
        self.transform = nn.Conv3d(out_ch, in_ch, kernel_size=kernel,stride=stride,padding=padding2)
    def forward(self,x):
        # h1 = self.bnorm1(self.relu(self.conv1(x)))
        # h = self.bnorm2(self.relu(self.conv2(h1)))
        h1 = self.relu(self.bnorm1(self.conv1(x)))
        h = self.relu(self.bnorm2(self.conv2(h1)))
        out = self.transform(h)
        # out = self.transform(h1)
        return out

class block2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=(1,1), padding=(1,1),padding1=(1,1),padding2=(0,0)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel,stride=stride,padding=padding1)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.transform = nn.Conv2d(out_ch, in_ch, kernel_size=kernel,stride=stride,padding=padding2)
    def forward(self,x):
        # h1 = self.bnorm1(self.relu(self.conv1(x)))
        # h = self.bnorm2(self.relu(self.conv2(h1)))
        h1 = self.relu(self.bnorm1(self.conv1(x)))
        h = self.relu(self.bnorm2(self.conv2(h1)))
        return self.transform(h)  

if __name__ == "__main__":
    model = block3d(in_ch=1,out_ch=64)
    # t = torch.full((1,), 100,  dtype=torch.long)
    a = torch.randn((100,1,104,16,16))
    print(model(a))