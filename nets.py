import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionNet(nn.Module):
    def __init__(self, ks = (35,3), nc = 64):
        super().__init__()
        self.name = 'DiffusionNet_'+str(ks[0])+'x'+str(ks[1])+'_'+str(nc)
        self.layers = nn.Sequential(
            nn.Conv2d(2, nc, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 2*nc, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nc, 4*nc, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*nc, 2*nc, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nc, nc, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 2, kernel_size=ks, stride=(1,1), padding=(ks[0]//2,ks[1]//2)),
            )

    def forward(self, x):
        x = self.layers(x)
        return x

## start UNET ->
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2,1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.name = 'Unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class DiffusionNet_compr(nn.Module):    # !! the funnel net might affect the spectral dimension (no pts)
    def __init__(self, ks1=(33,7), ks2=(34,7), nc=32):
        super().__init__()
        if ks2[1]%2==0:
            b_width = ks2[1]//2+1
        else:
            b_width = ks2[1]//2
        self.name = 'DiffusionNet_compr_'+str(ks1[0])+'x'+str(ks1[1])+'_'+str(ks2[0])+'x'+str(ks2[1])+'_'+str(nc)
        self.down = nn.Sequential(
            nn.Conv2d(2, nc, kernel_size=ks2, stride=(2,1),   padding=(ks2[0],    ks2[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 2*nc, kernel_size=ks1, stride=(1,1), padding=(ks1[0]//2, ks1[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nc, 4*nc, kernel_size=ks2, stride=(2,1), padding=(ks2[0],    ks2[1]//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*nc, 8*nc, kernel_size=ks1, stride=(1,1), padding=(ks1[0]//2, ks1[1]//2)),
            nn.ReLU(inplace=True),
            )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(8*nc, 4*nc, kernel_size=ks2, stride=(2,1), padding=(ks2[0],    b_width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*nc, 2*nc, kernel_size=ks1, stride=(1,1),          padding=(ks1[0]//2, ks1[1]//2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*nc, nc, kernel_size=ks2, stride=(2,1),   padding=(ks2[0],    b_width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 2, kernel_size=ks1, stride=(1,1),               padding=(ks1[0]//2, ks1[1]//2)),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x

# <- end unet
    
# ----------------------------------
#  TEST NEURAL NET ON RANDOM TENSOR:
# ----------------------------------
# import torch
# device = torch.device('cuda')
# #cnn = DiffusionNet(ks = (33,5), nc = 32).to(device)
# cnn = UNet(n_classes=2,n_channels=2).to(device)
# x = torch.randn((8, 2, 1000, 1)).to(device)
# print(x.shape)
# y = cnn(x)
# print(y.shape)