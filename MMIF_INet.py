from model import *
from invblock import INV_block
import modules.Unet_common as common
import torch.nn as nn
import torch


# three layer WT
class INet(nn.Module):

    def __init__(self, in_c1=3, in_c2=3, block_num=[2, 1, 1]):
        super(INet, self).__init__()
        self.in_c1 = in_c1
        self.in_c2 = in_c2

        self.dwt = common.DWT()
        self.iwt = common.IWT()
        inv_ops = []
        for i in range(block_num[0]):
            inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
        self.inv1_ops = nn.ModuleList(inv_ops)
        inv_ops = []
        for i in range(block_num[1]):
            inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
        self.inv2_ops = nn.ModuleList(inv_ops)
        inv_ops = []
        for i in range(block_num[2]):
            inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
        self.inv3_ops = nn.ModuleList(inv_ops)


        self.cat_ll1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        self.cat_ll2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        self.cat_ll3 = nn.Conv2d(in_c1*2, in_c1, 1, 1, 0)
        self.cat_lh3 = nn.Conv2d(in_c1*2, in_c1, 1, 1, 0)
        self.cat_hl3 = nn.Conv2d(in_c1*2, in_c1, 1, 1, 0)
        self.cat_hh3 = nn.Conv2d(in_c1*2, in_c1, 1, 1, 0)

        self.up_conv1 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv2 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv3 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.bn = nn.BatchNorm2d(in_c1)

    def forward(self, x, rev=False):
        # cover, secret  [B,C,H,W] torch.Size([4, 3, 128, 128])
        x1, x2 = (x.narrow(1, 0, self.in_c1), x.narrow(1, self.in_c1, self.in_c2))

        if not rev:
            # 先下采样
            # [B, C, H/2, W/2]
            x1_LL1, x1_HL1, x1_LH1, x1_HH1 = self.dwt(x1) # torch.Size([4, 3, 64, 64])
            x2_LL1, x2_HL1, x2_LH1, x2_HH1 = self.dwt(x2)
            for op in self.inv1_ops:
                x1_LL1, x2_LL1 = op.forward(torch.cat((x1_LL1, x2_LL1), dim=1))
            # [B, C, H / 4, W / 4]
            x1_LL2, x1_HL2, x1_LH2, x1_HH2 = self.dwt(x1_LL1)
            x2_LL2, x2_HL2, x2_LH2, x2_HH2 = self.dwt(x2_LL1)
            for op in self.inv2_ops:
                x1_LL2, x2_LL2 = op.forward(torch.cat((x1_LL2, x2_LL2), dim=1))
            # [B, C, H / 8, W / 8]
            x1_LL3, x1_HL3, x1_LH3, x1_HH3 = self.dwt(x1_LL2)
            x2_LL3, x2_HL3, x2_LH3, x2_HH3 = self.dwt(x2_LL2)
            for op in self.inv3_ops:
                x1_LL3, x2_LL3 = op.forward(torch.cat((x1_LL3, x2_LL3), dim=1))

            # Layer3 hight fre fusion
            matrix_w_ll = torch.sigmoid(self.cat_ll1(torch.cat((x1_LL3, x2_LL3), dim=1)))
            x_LL3 = matrix_w_ll * x1_LL3 + (1 - matrix_w_ll) * x2_LL3
            matrix_w_hl = torch.sigmoid(self.cat_hl1(torch.cat((x1_HL3, x2_HL3), dim=1)))
            x_HL3 = matrix_w_hl * x1_HL3 + (1-matrix_w_hl) * x2_HL3
            matrix_w_lh = torch.sigmoid(self.cat_lh1(torch.cat((x1_LH3, x2_LH3), dim=1)))
            x_LH3 = matrix_w_lh * x1_LH3 + (1 - matrix_w_lh) * x2_LH3
            matrix_w_hh = torch.sigmoid(self.cat_hh1(torch.cat((x1_HH3, x2_HH3), dim=1)))
            x_HH3 = matrix_w_hh * x1_HH3 + (1 - matrix_w_hh) * x2_HH3

            x_3 = self.iwt(torch.cat((x_LL3, x_HL3, x_LH3, x_HH3), dim=1))

            # Layer2 hight fre fusion
            matrix_w_ll = torch.sigmoid(self.cat_ll2(torch.cat((x1_LL2, x2_LL2), dim=1)))
            x_LL2 = matrix_w_ll * x1_LL2 + (1 - matrix_w_ll) * x2_LL2
            matrix_w_hl = torch.sigmoid(self.cat_hl2(torch.cat((x1_HL2, x2_HL2), dim=1)))
            x_HL2 = matrix_w_hl * x1_HL2 + (1 - matrix_w_hl) * x2_HL2
            matrix_w_lh = torch.sigmoid(self.cat_lh2(torch.cat((x1_LH2, x2_LH2), dim=1)))
            x_LH2 = matrix_w_lh * x1_LH2 + (1 - matrix_w_lh) * x2_LH2
            matrix_w_hh = torch.sigmoid(self.cat_hh2(torch.cat((x1_HH2, x2_HH2), dim=1)))
            x_HH2 = matrix_w_hh * x1_HH2 + (1 - matrix_w_hh) * x2_HH2

            x_2 = self.iwt(torch.cat((x_LL2, x_HL2, x_LH2, x_HH2), dim=1))

            # Layer1 hight fre fusion
            matrix_w_ll = torch.sigmoid(self.cat_ll3(torch.cat((x1_LL1, x2_LL1), dim=1)))
            x_LL1 = matrix_w_ll * x1_LL1 + (1 - matrix_w_ll) * x2_LL1
            matrix_w_hl = torch.sigmoid(self.cat_hl3(torch.cat((x1_HL1, x2_HL1), dim=1)))
            x_HL1 = matrix_w_hl * x1_HL1 + (1 - matrix_w_hl) * x2_HL1
            matrix_w_lh = torch.sigmoid(self.cat_lh3(torch.cat((x1_LH1, x2_LH1), dim=1)))
            x_LH1 = matrix_w_lh * x1_LH1 + (1 - matrix_w_lh) * x2_LH1
            matrix_w_hh = torch.sigmoid(self.cat_hh3(torch.cat((x1_HH1, x2_HH1), dim=1)))
            x_HH1 = matrix_w_hh * x1_HH1 + (1 - matrix_w_hh) * x2_HH1

            # 图像重构
            x_1 = self.iwt(torch.cat((x_LL1, x_HL1, x_LH1, x_HH1), dim=1))

            out = self.up_conv1(self.upsample(x_3)) + x_2
            out = self.up_conv2(self.upsample(out)) + x_1
            out = self.bn(out)

        # return out, x1_LL1, x_LL1, x2_LL1
        return out