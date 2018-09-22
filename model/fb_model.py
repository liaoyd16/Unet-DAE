
'''imports'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np


def odd(w):
    return list(np.arange(1, w, step=2, dtype='long'))

def even(w):
    return list(np.arange(0, w, step=2, dtype='long'))


''' ResBlock '''
class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlock, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

    def forward(self, x):
        if self.channels_out > self.channels_in:
            x1 = F.relu(self.conv1(x))
            x1 =        self.conv2(x1)
            x  = self.sizematch(self.channels_in, self.channels_out, x)
            return x + x1
        elif self.channels_out < self.channels_in:
            x = F.relu(self.conv1(x))
            x1 =       self.conv2(x)
            x = x + x1
            return x
        else:
            x1 = F.relu(self.conv1(x))
            x1 =        self.conv2(x1)
            x = x + x1
            return x

    def sizematch(self, channels_in, channels_out, x):
        zeros = torch.zeros( (x.size()[0], channels_out - channels_in, x.shape[2], x.shape[3]), dtype=torch.float )
        return torch.cat((x, zeros), dim=1)

class ResTranspose(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResTranspose, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(2,2), stride=2)
        self.deconv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

    def forward(self, x):
        # cin = cout
        x1 = F.relu(self.deconv1(x))
        x1 =        self.deconv2(x1)
        x = self.sizematch(x)
        return x + x1

    def sizematch(self, x):
        # expand
        x2 = torch.zeros(x.shape[0], self.channels_in, x.shape[2]*2, x.shape[3]*2)

        row_x  = torch.zeros(x.shape[0], self.channels_in, x.shape[2], 2*x.shape[3])
        row_x[:,:,:,odd(x.shape[3]*2)]   = x
        row_x[:,:,:,even(x.shape[3]*2)]  = x
        x2[:,:, odd(x.shape[2]*2),:] = row_x
        x2[:,:,even(x.shape[2]*2),:] = row_x

        return x2


def initialize(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
    if isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight)



class ResDAE(nn.Module):
    def __init__(self):
        super(ResDAE, self).__init__()

        self.upward_net1 = nn.Sequential(
            # 128x128x8
            ResBlock(1, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8),
        )
        self.upward_net2 = nn.Sequential(
            # 64x64x16
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16),
        )
        self.upward_net3 = nn.Sequential(
            # 32x32x32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.BatchNorm2d(32),
        )
        self.upward_net4 = nn.Sequential(
            # 16x16x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )
        self.upward_net5 = nn.Sequential(
            # 8x8x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )

        self.linear1 = nn.Linear(4096, 512)

        self.linear2 = nn.Linear(512, 4096)

        self.uconv5 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.downward_net5 = nn.Sequential(
            # 8x8x64
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResTranspose(64, 64),
            nn.BatchNorm2d(64),
        )
        self.uconv4 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.downward_net4 = nn.Sequential(
            # 16x16x64
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResTranspose(32, 32),
            nn.BatchNorm2d(32),
        )
        self.uconv3 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=(1,1))
        self.downward_net3 = nn.Sequential(
            # 32x32x32
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResTranspose(16, 16),
            nn.BatchNorm2d(16),
        )
        self.uconv2 = nn.Conv2d(32, 16, kernel_size=(3,3), padding=(1,1))
        self.downward_net2 = nn.Sequential(
            # 64x64x16
            ResBlock(16, 16),
            ResBlock(16, 8),
            ResBlock(8, 8),
            ResTranspose(8, 8),
            nn.BatchNorm2d(8),
        )
        # self.uconv1 = nn.Conv2d(16, 8, kernel_size=(3,3), padding=(1,1))
        self.downward_net1 = nn.Sequential(
            # 128x128x8
            ResBlock(8, 8),
            ResBlock(8, 4),
            ResBlock(4, 1),
            ResBlock(1, 1),
            nn.BatchNorm2d(1),
        )


    def upward(self, x, a5=None, a4=None, a3=None, a2=None, a1=None):
        bs = x.shape[0]
        x = x.view(bs, 1, 128, 128)

        x = self.upward_net1(x)
        self.x1 = x

        x = self.upward_net2(x)         # 8x64x64
        if a1 is not None: x = x * a1   
        self.x2 = x
        
        x = self.upward_net3(x)         # 16x32x32
        if a2 is not None: x = x * a2
        self.x3 = x
        
        x = self.upward_net4(x)         # 32x16x16
        if a3 is not None: x = x * a3
        self.x4 = x
        
        x = self.upward_net5(x)         # 64x8x8
        if a4 is not None: x = x * a4
        self.x5 = x

        x = x.view(bs, 4096)
        x = F.relu(self.linear1(x))
        if a5 is not None: x = x * a5

        self.top = x

        return x


    def downward(self, y, shortcut=True):

        bs = y.shape[0]
        y = y.view(bs, 512)

        y = F.relu(self.linear2(y))
        y = y.view(bs, 64, 8, 8)

        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)
        
        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)
        
        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)
        
        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)
        
        # if shortcut:
        #     y = torch.cat((y, self.x1), 1)
        #     y = F.relu(self.uconv1(y))
        y = self.downward_net1(y)

        return y



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(4096, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 4096)

        x = self.linear1(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x


class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()

        self.linear5 = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 2)

        a5 = self.linear5(x).view(bs, 512)
        a4 = self.linear4(x).view(bs, 64, 1, 1)
        a3 = self.linear3(x).view(bs, 64, 1, 1)
        a2 = self.linear2(x).view(bs, 32, 1, 1)
        a1 = self.linear1(x).view(bs, 16, 1, 1)

        return a5, a4, a3, a2, a1
