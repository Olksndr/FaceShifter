import torch
from itertools import chain

class ConvUnit(torch.nn.Module):
    def __init__(self, input_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.conv = torch.nn.Conv2d(input_channels, out_channels, (4, 4), 2, 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu_op = torch.nn.LeakyReLU(0.1)

    def forward(self, input_x):
        conv_out = self.conv(input_x)
        batch_norm_out = self.batch_norm(conv_out)
        output = self.relu_op(batch_norm_out)

        return output

class ConvTransposeUnit(torch.nn.Module):
    def __init__(self, input_channels, out_channels):
        super(ConvTransposeUnit, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(input_channels, out_channels, (4, 4), 2, 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu_op = torch.nn.LeakyReLU(0.1)

    def forward(self, input_x):
        conv_out = self.conv(input_x)
        batch_norm_out = self.batch_norm(conv_out)
        output = self.relu_op(batch_norm_out)

        return output

class MultiLevelAttEncoder(torch.nn.Module):
    """Please refer Fig 12 https://arxiv.org/pdf/1912.13457.pdf"""
    def __init__(self):
        super(MultiLevelAttEncoder, self).__init__()

        self.down_conv_0 = ConvUnit(3, 32)
        self.down_conv_1 = ConvUnit(32, 64)
        self.down_conv_2 = ConvUnit(64, 128)
        self.down_conv_3 = ConvUnit(128, 256)
        self.down_conv_4 = ConvUnit(256, 512)
        self.down_conv_5 = ConvUnit(512, 1024)
        self.down_conv_6 = ConvUnit(1024, 1024)

        self.up_conv_0 = ConvTransposeUnit(1024, 1024)
        self.up_conv_1 = ConvTransposeUnit(2048, 512)
        self.up_conv_2 = ConvTransposeUnit(1024, 256)
        self.up_conv_3 = ConvTransposeUnit(512, 128)
        self.up_conv_4 = ConvTransposeUnit(256, 64)
        self.up_conv_5 = ConvTransposeUnit(128, 32)

    def forward(self, input_x):
        x_1_d = self.down_conv_0(input_x)
        x_2_d = self.down_conv_1(x_1_d)
        x_3_d = self.down_conv_2(x_2_d)
        x_4_d = self.down_conv_3(x_3_d)
        x_5_d = self.down_conv_4(x_4_d)
        x_6_d = self.down_conv_5(x_5_d)
        z_1_att = self.down_conv_6(x_6_d)

        x_1_up = self.up_conv_0(z_1_att)

        z_2_att = torch.cat([x_1_up, x_6_d], dim=1)
        x_2_up = self.up_conv_1(z_2_att)

        z_3_att = torch.cat([x_2_up, x_5_d], dim=1)
        x_3_up = self.up_conv_2(z_3_att)

        z_4_att = torch.cat([x_3_up, x_4_d], dim=1)
        x_4_up = self.up_conv_3(z_4_att)

        z_5_att = torch.cat([x_4_up, x_3_d], dim=1)
        x_5_up = self.up_conv_4(z_5_att)

        z_6_att = torch.cat([x_5_up, x_2_d], dim=1)
        x_6_up = self.up_conv_5(z_6_att)

        z_7_att = torch.cat([x_6_up, x_1_d], dim=1)

        z_8_att = torch.nn.UpsamplingBilinear2d(scale_factor=2)(z_7_att)


        return z_1_att, z_2_att, z_3_att, z_4_att, z_5_att, z_6_att, z_7_att, z_8_att
