# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
from re import X

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class PersonalizedChannelSelection(nn.Module):
    def __init__(self, f_dim, emb_dim):
        super(PersonalizedChannelSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(nn.Conv2d(emb_dim, f_dim, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(f_dim, f_dim, 1, bias=False))
        self.fc2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(f_dim // 16, f_dim, 1, bias=False))

    def forward_emb(self, emb):
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        emb = self.fc1(emb)
        return emb

    def forward(self, x, emb):
        b, c, w, h = x.size()

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # site embedding
        emb = self.forward_emb(emb)

        # avg
        avg_out = torch.cat([avg_out, emb], dim=1)
        avg_out = self.fc2(avg_out)

        # max
        max_out = torch.cat([max_out, emb], dim=1)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        hmap = self.sigmoid(out)

        x = x * hmap + x

        return x, hmap

class LCEncoder(nn.Module):
    def __init__(self, params):
        super(LCEncoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.n_pcs = self.params['pcs_num']
        self.n_emb = self.params['emb_num']
        self.n_client = self.params['client_num']
        self.cid = self.params['client_id']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.conv_list = [self.in_conv, self.down1, self.down2, self.down3, self.down4]

        self.pcs_list = []
        for i in range(self.n_pcs): 
            # print('PCS', (2**(5 - self.n_pcs + i)) * 16, self.ft_chns[5 - self.n_pcs + i])
            self.pcs_list.append(
                PersonalizedChannelSelection(self.ft_chns[5 - self.n_pcs + i], self.n_emb).cuda()
            )

    def forward(self, x, emb_idx=None):
        def get_emb(emb_idx):
            batch_size = x.size(0)
            emb = torch.zeros((batch_size, self.n_client)).cuda()
            emb[:, emb_idx] = 1
            return emb

        if not emb_idx:
            emb = get_emb(self.cid)
        else:
            emb = get_emb(emb_idx)

        heatmaps = []
        features = []
        for i in range(len(self.conv_list)):
            # print(i, x.shape, self.conv_list[i])
            x = self.conv_list[i](x)
            if i >= (len(self.conv_list) - self.n_pcs):
                x, heatmap = self.pcs_list[i - len(self.conv_list) + self.n_pcs](x, emb)
            else:
                heatmap = None
            features.append(x)
            heatmaps.append(heatmap)

        return features, heatmaps


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        return output, x_1, x_2, x_3, x_4


class Decoder_Head(nn.Module):
    def __init__(self, params):
        super(Decoder_Head, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output = self.dsn_head(x_2)
        return output, x_1, x_2, x_3, x_4, aux_output


class Decoder_MultiHead(nn.Module):
    def __init__(self, params):
        super(Decoder_MultiHead, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head1 = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head2 = nn.Sequential(
            nn.Conv2d(self.ft_chns[1], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head3 = nn.Sequential(
            nn.Conv2d(self.ft_chns[0], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output1 = self.dsn_head1(x_2)
        aux_output2 = self.dsn_head2(x_3)
        aux_output3 = self.dsn_head3(x_4)
        return output, x_1, x_2, x_3, x_4, aux_output1, aux_output2, aux_output3

class Decoder_MultiHead_Two(nn.Module):
    def __init__(self, params):
        super(Decoder_MultiHead_Two, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.dsn_head1 = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head2 = nn.Sequential(
            nn.Conv2d(self.ft_chns[1], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_1 = self.up1(x4, x3)
        x_2 = self.up2(x_1, x2)
        x_3 = self.up3(x_2, x1)
        x_4 = self.up4(x_3, x0)
        output = self.out_conv(x_4)
        aux_output1 = self.dsn_head1(x_2)
        aux_output2 = self.dsn_head2(x_3)
        return output, x_1, x_2, x_3, x_4, aux_output1, aux_output2


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4]


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)[0]
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)[0]
        return main_seg, aux_seg1


class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2


class UNet_Head(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Head, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_Head(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4, aux_output = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, aux_output]


class UNet_MultiHead(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_MultiHead, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_MultiHead(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3 = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3]
    


class UNet_LC(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_Head(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap,aux_output]

class UNet_LC_MultiHead(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC_MultiHead, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_MultiHead(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap, aux_output1, aux_output2, aux_output3]



class UNet_LC_MultiHead_Two(nn.Module):
    def __init__(self, in_chns, class_num, pcs_num, emb_num, client_num, client_id):
        super(UNet_LC_MultiHead_Two, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'pcs_num': pcs_num,
                  'emb_num': emb_num,
                  'client_num': client_num,
                  'client_id': client_id}

        self.encoder = LCEncoder(params)
        self.decoder = Decoder_MultiHead_Two(params)

    def forward(self, x, emb_idx=None):
        feature, heatmap = self.encoder(x, emb_idx)
        output, de1, de2, de3, de4, aux_output1, aux_output2  = self.decoder(feature)
        return [output, feature, de1, de2, de3, de4, heatmap, aux_output1, aux_output2]
