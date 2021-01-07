from torch import nn

import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils import spectral_norm


# Reshape + Concat layer

class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        # super(Reshape_Concat_Adap, self).__init__()
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                # data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                # print data_temp.shape
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                # data_temp = torch.zeros(data_t.shape).cuda() + data_t
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


class SamplingModule(nn.Module):
    def __init__(self,group_num,sample_channel,blocksize):
        super(SamplingModule,self).__init__()
        self.group_num = group_num
        for m in range(group_num):
            sampleblock=nn.Conv2d(1,sample_channel, blocksize, stride=blocksize, padding=0, bias=False)
            self.add_module(f'sampleblock_{str(m)}', sampleblock) 

    def forward(self,x):
        samplelist = []
        for m in range(self.group_num):
            sampleblock = getattr(self, f'sampleblock_{str(m)}')
            samplelist.append(sampleblock(x))
        return samplelist

class HeadBlock(nn.Module):
    def __init__(self,sample_channel,out_channel,blocksize):
        super(HeadBlock, self).__init__()
        self.blocksize = blocksize
        self.upsampling = nn.Conv2d(sample_channel, out_channel, 1, stride=1, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self,x):
        x = self.upsampling(x)
        x = My_Reshape_Adap(x, self.blocksize)
        block1 = self.conv1(x)
        return block1

class ToOutput(nn.Module):
    def __init__(self):
        super(ToOutput, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self,block1):
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)
        return block5

class Variance_estimation_block(nn.Module):
    def __init__(self,mode="concate"):
        super(Variance_estimation_block, self).__init__()
        
        self.fusion_block=HierarchicalBlock(mode=mode)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self,x,y):
        block1 = self.fusion_block(x,y)
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)
        return block5


class HierarchicalBlock(nn.Module):
    def __init__(self,mode="concate"):
        super(HierarchicalBlock, self).__init__()
        if mode == "concate":
            FusionBlock = Fusion_concate
        elif mode == "spade":
            FusionBlock = Fusion_spade
        elif mode == "spade_res":
            FusionBlock = Fusion_spade_residual
        elif mode == "add":
            FusionBlock = Fusion_add
        elif mode == "add_res":
            FusionBlock = Fusion_add_residual

        
        self.fusion_block = FusionBlock()
    
    def forward(self,x,y):
        return self.fusion_block(x,y)


class Fusion_concate(nn.Module):
    def __init__(self,in_chan=128,num_filters=64):
        super(Fusion_concate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, num_filters, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self,x,y):
        return self.conv(torch.cat((x,y),dim=1))

class Fusion_add(nn.Module):
    def __init__(self):
        super(Fusion_add, self).__init__()

    def forward(self,x,y):
        return torch.add(x, y)

class Fusion_add_residual(nn.Module):
    def __init__(self,in_chan=64):
        super(Fusion_add_residual, self).__init__()
        self.normx = nn.BatchNorm2d(in_chan)
        self.normy = nn.BatchNorm2d(in_chan)
        self.activation = nn.PReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_chan, 2*in_chan, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.norm_concate = nn.BatchNorm2d(2*in_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*in_chan, in_chan, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
    def forward(self,x,y):
        identity_map = x
        resx = self.normx(x)
        resx = self.activation(resx)
        resy = self.normy(y)
        resy = self.activation(resy)
        res = self.conv1(torch.cat((resx,resy),dim=1))
        res = self.norm_concate(res)
        res = self.activation(res)
        res = self.conv2(res)
        return torch.add(x,res)



class Fusion_spade(nn.Module):
    def __init__(self,in_chan=64,num_filters=64):
        super(Fusion_spade, self).__init__()
        self.activation = nn.PReLU()
        self.conv = spectral_norm(nn.Conv2d(in_chan, num_filters, kernel_size=(3, 3), padding=1))
        self.conv_gamma = spectral_norm(nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3), padding=1))
        self.conv_beta = spectral_norm(nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3), padding=1))
    
    def forward(self,x,seg):
        seg = self.activation(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)
        x = x*(1+seg_gamma) + seg_beta
        return x

class Fusion_spade_residual(nn.Module):
    def __init__(self,in_chan=64,num_filters=64,kernel_size=3, stride=1):
        super(Fusion_spade_residual, self).__init__()
        self.activation = nn.PReLU()
        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_chan, num_filters, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_chan, num_filters, kernel_size, stride=stride)
        self.norm1 = nn.BatchNorm2d(in_chan)
        self.norm2 = nn.BatchNorm2d(in_chan)
        self.spade1 = Fusion_spade()
        self.spade2 = Fusion_spade()
    
    def forward(self,x,style):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res)
        res = self.spade1(res,style)
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)
        res = self.spade2(res,style)

        return torch.add(res, identity_map)

#  code of CSNet
class HierarchicalCSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1, group_num=8,mode="concate",shared_head=False,shared_tail=False,variance_estimation=False):

        super(HierarchicalCSNet, self).__init__()
        self.blocksize = blocksize
        self.group_num = group_num
        # for sampling
        self.sampling = SamplingModule(group_num=group_num,sample_channel=int(np.round(blocksize*blocksize*subrate/group_num)),blocksize=blocksize)
        # self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # upsampling
        # self.upsampling = nn.Conv2d(int(np.round(blocksize*blocksize*subrate)), blocksize*blocksize, 1, stride=1, padding=0)
        if shared_head == False:
            for m in range(group_num):
                headblock = HeadBlock(
                    sample_channel=int(np.round(blocksize*blocksize*subrate/group_num)),
                    out_channel = blocksize*blocksize,
                    blocksize = blocksize
                )
                self.add_module(f'head_{str(m)}', headblock)
        else:
            headblock = HeadBlock(
                    sample_channel=int(np.round(blocksize*blocksize*subrate/group_num)),
                    out_channel = blocksize*blocksize,
                    blocksize = blocksize
                )
            for m in range(group_num):
                self.add_module(f'head_{str(m)}', headblock)
        

        for m in range(group_num-1):
            hierarchicalblock = HierarchicalBlock(mode=mode)
            self.add_module(f'hierarchicalblock_{str(m)}', hierarchicalblock)

        self.variance_estimation = variance_estimation
        if variance_estimation == True:
            for m in range(group_num-1):
                varianceestimationblock = Variance_estimation_block(mode=mode)
                self.add_module(f'varianceestimationblock_{str(m)}', varianceestimationblock)
        
        if shared_tail == False:
            for m in range(group_num):
                tailblock = ToOutput()
                self.add_module(f'tail_{str(m)}', tailblock)
        else:
            tailblock = ToOutput()
            for m in range(group_num):
                self.add_module(f'tail_{str(m)}', tailblock)


    def forward(self, x):
        y = self.sampling(x)
        headlist = []
        for m in range(self.group_num):
            headlist.append(getattr(self, f'head_{str(m)}')(y[m]))

        resultlist = []
        x = headlist[0]
        resultlist.append(getattr(self, f'tail_{str(0)}')(x))

        if self.variance_estimation == True:
            variance_estimation_list = []

        for m in range(self.group_num-1):
            x = getattr(self, f'hierarchicalblock_{str(m)}')(x,headlist[m+1])
            if self.variance_estimation == True:
                variance_estimation_list.append(getattr(self, f'varianceestimationblock_{str(m)}')(x,headlist[m+1]))
            resultlist.append(getattr(self, f'tail_{str(m+1)}')(x))

        if self.variance_estimation == True:
            return resultlist, variance_estimation_list
        else:
            return resultlist
