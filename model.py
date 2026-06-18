
import torch.nn.functional as F
import torch
import torch.nn as nn
from scipy import io as sio
import numpy as np



# cpu_num是一个整数
torch.set_num_threads(2)

class Encoder_Base(nn.Module):
    def __init__(self, args):
        super(Encoder_Base, self).__init__()
        Layernum = args.Layernum1  # [channel, 32, num_em]
        self.num_em = args.num_em
        self.conv11 = nn.Sequential(conv1x1(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))
        self.conv12 = nn.Sequential(conv3x3(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))

        self.conv2 = nn.Sequential(conv1x1(2 * Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)

        self.conv3 = nn.Sequential(conv1x1(Layernum[1], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)

        self.conv4 = nn.Sequential(conv1x1(Layernum[2], Layernum[3]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[3]))
        self.sumtoone = SumToOne(args)

    def forward(self, x):
        x1 = self.conv11(x)  # [batchsize, C, height, width]
        x2 = self.conv12(x)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv2(x)
        x = self.spatialAtt1(x)

        x = self.conv3(x)
        fea = self.spatialAtt2(x)

        x = self.conv4(fea)

        x = self.sumtoone(x)
        return fea, x  # [B, out_planes, H,W]

    def getBasic(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x
    def getSpatialAtt1(self,x):
        x = self.getBasic(x)
        x = self.spatialAtt1(x)
        return x
    def getSpatialAtt2(self,x):
        x = self.getSpatialAtt1(x)
        x = self.conv3(x)
        x = self.spatialAtt2(x)
        return x
class Encoder_FourLayer(nn.Module):
    def __init__(self, args):
        super(Encoder_FourLayer, self).__init__()
        Layernum = args.Layernum1  # [channel, 32, num_em]
        self.num_em = args.num_em
        self.conv11 = nn.Sequential(conv1x1(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))
        self.conv12 = nn.Sequential(conv3x3(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))

        self.conv2 = nn.Sequential(conv1x1(2 * Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)

        self.conv2_2 = nn.Sequential(conv3x3( Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1_2 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)



        self.conv3 = nn.Sequential(conv1x1(Layernum[1], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)
        self.conv3_2 = nn.Sequential(conv3x3(Layernum[2], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2_2 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)




        self.conv4 = nn.Sequential(conv1x1(Layernum[2], Layernum[3]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[3]))
        self.sumtoone = SumToOne(args)

    def forward(self, x):
        x1 = self.conv11(x)  # [batchsize, C, height, width]
        x2 = self.conv12(x)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv2(x)
        x = self.spatialAtt1(x)
        x = self.conv2_2(x)
        x = self.spatialAtt1_2(x)

        x = self.conv3(x)
        x = self.spatialAtt2(x)
        x = self.conv3_2(x)
        fea = self.spatialAtt2_2(x)

        x = self.conv4(fea)

        x = self.sumtoone(x)
        return fea, x  # [B, out_planes, H,W]

    def getBasic(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x
    def getSpatialAtt1(self,x):
        x = self.getBasic(x)
        x = self.spatialAtt1(x)
        return x
    def getSpatialAtt2(self,x):
        x = self.getSpatialAtt1(x)
        x = self.conv3(x)
        x = self.spatialAtt2(x)
        return x
class Encoder_SixLayer(nn.Module):
    def __init__(self, args):
        super(Encoder_SixLayer, self).__init__()
        Layernum = args.Layernum1  # [channel, 32, num_em]
        self.num_em = args.num_em
        self.conv11 = nn.Sequential(conv1x1(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))
        self.conv12 = nn.Sequential(conv3x3(Layernum[0], Layernum[1]),
                                    nn.LeakyReLU(0.02),
                                    nn.BatchNorm2d(Layernum[1]))

        self.conv2 = nn.Sequential(conv1x1(2 * Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)

        self.conv2_2 = nn.Sequential(conv3x3( Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1_2 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)
        self.conv2_3 = nn.Sequential(conv3x3( Layernum[1], Layernum[1]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[1]))
        self.spatialAtt1_3 = MultiHead_SpatialAttention(in_channel=Layernum[1], num_em=args.num_em)



        self.conv3 = nn.Sequential(conv1x1(Layernum[1], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)
        self.conv3_2 = nn.Sequential(conv3x3(Layernum[2], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2_2 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)
        self.conv3_3 = nn.Sequential(conv3x3(Layernum[2], Layernum[2]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[2]))
        self.spatialAtt2_3 = MultiHead_SpatialAttention(in_channel=Layernum[2], num_em=args.num_em)




        self.conv4 = nn.Sequential(conv1x1(Layernum[2], Layernum[3]),
                                   nn.LeakyReLU(0.02),
                                   nn.BatchNorm2d(Layernum[3]))
        self.sumtoone = SumToOne(args)

    def forward(self, x):
        x1 = self.conv11(x)  # [batchsize, C, height, width]
        x2 = self.conv12(x)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv2(x)
        x = self.spatialAtt1(x)
        x = self.conv2_2(x)
        x = self.spatialAtt1_2(x)
        x = self.conv2_3(x)
        x = self.spatialAtt1_3(x)


        x = self.conv3(x)
        x = self.spatialAtt2(x)
        x = self.conv3_2(x)
        fea = self.spatialAtt2_2(x)
        x = self.conv3_3(x)
        fea = self.spatialAtt2_3(x)

        x = self.conv4(fea)

        x = self.sumtoone(x)
        return fea, x  # [B, out_planes, H,W]

    def getBasic(self, x):
        x1 = self.conv11(x)
        x2 = self.conv12(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x
    def getSpatialAtt1(self,x):
        x = self.getBasic(x)
        x = self.spatialAtt1(x)
        return x
    def getSpatialAtt2(self,x):
        x = self.getSpatialAtt1(x)
        x = self.conv3(x)
        x = self.spatialAtt2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args,
                 dim,
                 qkv_PROJ=True,
                 qkv_bias=True,
                 stride=1,
                 padding=True,
                 kernel_size=3):
        super().__init__()
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = 1
        self.head_channel = dim
        # it seems that padding must be true to make unfolded dim matchs query dim h*w*ks*ks
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = dim ** -0.5
        self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
        self.qkv_PROJ = qkv_PROJ
        if qkv_PROJ:
            self.qkv_bias = qkv_bias
            self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
            self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        else:
            self.q_proj = None
            self.k_proj = None

        self.softmax = nn.Softmax(dim=-1)
        w1 = torch.randn(self.k_size*self.k_size, args.num_em, args.channel)
        nn.init.kaiming_normal_(w1, mode='fan_out')
        self.decoder_weight = nn.Parameter(data=w1, requires_grad=True)  # [ks*ks, num_em,channel]
        self.relu = nn.ReLU()

    def padding(self,k):
        # B,_, _, H, W = k.shape   # (B, NumHeads, HeadsC, H, W)
        B, NumHeads, HeadsC, H, W = k.shape  # (B, NumHeads, HeadsC, H, W)
        k = self.pad(k)  # (B, NumH, HeadC, H, W)
        k = k.permute(0, 1, 3, 4, 2)  # (B, NumHeads, H, W, HeadC)
        H, W = H + self.pad_size * 2, W + self.pad_size * 2
        # unfold plays role of conv2d to get patch data
        k = k.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        k = self.unfold(k)
        k = k.reshape(B, NumHeads, HeadsC, self.k_size ** 2,
                      self.num_patch)  # (B, NumH, HC, ks*ks, NumPatch)
        k = k.permute(0, 1, 4, 3, 2)  # (B, NumH, NumPatch, ks*ks, HC)
        return k

    def forward(self, x, abund):
        B, C, H, W = x.shape
        assert C == self.in_channels
        self.pat_size_h = (H + 2 * self.pad_size - self.k_size) // self.stride + 1
        self.pat_size_w = (W + 2 * self.pad_size - self.k_size) // self.stride + 1
        self.num_patch = self.pat_size_h * self.pat_size_w
        if self.qkv_PROJ:
            # (B, NumHeads, H, W, HeadC)
            q = self.q_proj(x).reshape(B, self.num_heads, self.head_channel,H, W).permute(0,1,3,4,2)
            # (B, NumHeads, HeadsC, H, W)
            k = self.k_proj(x).reshape(B, self.num_heads, self.head_channel, H, W)
        else:
            q = x.reshape(B, self.num_heads, self.head_channel,H, W).permute(0,1,3,4,2)
            k = x.reshape(B, self.num_heads, self.head_channel, H, W)
        q = q.unsqueeze(dim=4)   # (B, NumHeads, H, W, 1, HeadC)
        q = q * self.scale
        k = self.padding(k)  # (B, NumH, NumPatch, ks*ks, HC)

        _, num_em, h, w = abund.shape
        assert h == H
        assert w == W
        abund = abund.unsqueeze(1) # (B, 1, num_em, h, w)
        abund = self.padding(abund)  # (B, NumH, NumPatch, ks*ks, num_em)
        abund = abund.squeeze(1).unsqueeze(3)  # (B, NumPatch, ks*ks, 1, num_em)

        # (B, NumH, NumPatch, 1, HeadC)
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
        att = (q @ k.transpose(-2, -1))  # (B, NumH=1, NumPatch, 1, ks*ks)
        att = self.softmax(att)  # softmax last dim, affinity map
        att = att.squeeze(1)  # [B, NumPatch, ks*ks]
        att = att.squeeze(2)
        att = att.unsqueeze(3).unsqueeze(3) # [B, NumPatch, ks*ks,1,1]
        self.refineded_decoder_weight = self.decoder_weight.unsqueeze(0).unsqueeze(0)  # [1,1,ks*ks, num_em,channel]
        # [B,NumPatch, ks*ks, num_em,channel]
        self.refineded_decoder_weight = self.refineded_decoder_weight.repeat(B, h * w, 1, 1, 1)
        # (B, NumPatch, ks*ks, num_em, channel)
        self.refineded_decoder_weight = att * self.refineded_decoder_weight

        recon_x = torch.matmul(abund, self.refineded_decoder_weight)  # [B, NumPatch, ks*ks,1,channel]
        recon_x = recon_x.sum(2)  # [B, NumPatch,1,channel]
        recon_x = recon_x.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # [B, channel, h,w]
        recon_x = self.relu(recon_x)
        return recon_x

    def getEndmembers(self):
        w = torch.clamp_min(self.decoder_weight, 0)  # 将参数范围限制到0-+之间
        return w
class ACRNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(ACRNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        fea, abunds = self.encoder(x)
        recon_x = self.decoder(fea,abunds)
        return abunds, recon_x
    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=0)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers
    def get_recon_x(self,x):
        fea, abunds = self.encoder(x)
        recon_x = self.decoder(fea, abunds)
        return recon_x
    def getAbundances(self, hsi):
        _, abunds = self.encoder(hsi)
        return abunds.squeeze()



# 常见的1x1卷积
def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(stride,stride),
                     padding=0, bias=False)
# 常见的3x3卷积
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=1, bias=False)
class SumToOne(nn.Module):
    def __init__(self, args):
        super(SumToOne, self).__init__()
        self.scale = args.scale
    def forward(self, x):
        x = F.softmax(self.scale * x, dim=1)
        return x
class MultiHead_SpatialAttention(nn.Module):
    def __init__(self, in_channel, num_em, kernel_size=7):
        super(MultiHead_SpatialAttention, self).__init__()
        assert in_channel//num_em, 'feature channel must be num_em*8'
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.num_em = num_em
        self.in_dim = int(in_channel // num_em)
        self.spatialAtt = nn.ModuleList(
            [nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=padding, bias=False) for i in range(self.num_em)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.split(x,self.in_dim, dim=1)
        Out = []
        for i in range(self.num_em):
            out = x[i]
            # out = self.proj[i](x[i])
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            fea = torch.cat([avg_out, max_out], dim=1)
            fea = self.spatialAtt[i](fea)
            out = out*self.sigmoid(fea)
            Out.append(out)
        out = torch.cat(Out,dim=1)
        return out
    def getSpatiMap(self,x):
        x = torch.split(x, self.in_dim, dim=1)
        Attmap = []
        for i in range(self.num_em):
            out = x[i]
            # out = self.proj[i](x[i])
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            fea = torch.cat([avg_out, max_out], dim=1)
            fea = self.spatialAtt[i](fea)
            SpatiMap = self.sigmoid(fea)
            Attmap.append(SpatiMap)
        return Attmap
