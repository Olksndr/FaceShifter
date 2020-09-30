import torch
from itertools import chain

class ConvUnit(torch.nn.Module):
    def __init__(self, channels):
        super(ConvUnit, self).__init__()
        input_channels, out_channels = channels
        self.conv = torch.nn.Conv2d(input_channels, out_channels, (4, 4), 2, 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu_op = torch.nn.LeakyReLU()

    def forward(self, input_x):
        conv_out = self.conv(input_x)
        batch_norm_out = self.batch_norm(conv_out)
        output = self.relu_op(batch_norm_out)

        return output

class ConvTransposeUnit(torch.nn.Module):
    def __init__(self, channels):
        super(ConvTransposeUnit, self).__init__()
        input_channels, out_channels = channels
        self.conv = torch.nn.ConvTranspose2d(input_channels, out_channels, (4, 4), 2, 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu_op = torch.nn.LeakyReLU()

    def forward(self, input_x):
        conv_out = self.conv(input_x)
        batch_norm_out = self.batch_norm(conv_out)
        output = self.relu_op(batch_norm_out)

        return output

class MultiLevelAttEncoder(torch.nn.Module):
    """Please refer Fig 12 https://arxiv.org/pdf/1912.13457.pdf"""
    def __init__(self, input_channels, output_channels):
        super(MultiLevelAttEncoder, self).__init__()
        self.fw_channels_generator = chain(zip(input_channels, output_channels))
        self.conv_units_fw = [*map(lambda channels: ConvUnit(channels), self.fw_channels_generator)]

        bw_input_channels = input_channels.copy()
        bw_output_channels = output_channels.copy()
        bw_input_channels.reverse()
        bw_output_channels.reverse()

        self.bw_channels_generator = chain(zip(bw_output_channels[:-1], bw_input_channels[:-1]))
        self.conv_units_bw = [*map(lambda channels: ConvTransposeUnit(channels), self.bw_channels_generator)]

    def fw_pass(self, input_x):
        #input_x = X_t [batch_size, 3, 256, 256]

        intermidiate_outputs = []
        for unit in self.conv_units_fw:
            out = unit(input_x)
            intermidiate_outputs += [out]
            input_x = out
        return intermidiate_outputs

    def bw_pass(self, intermidiate_outputs):

        z_att_0 = intermidiate_outputs.pop()
        intermidiate_outputs.reverse()

        att_features = [z_att_0]

        z_att_k = z_att_0
        for i, unit in enumerate(self.conv_units_bw):

            z_att_k = unit(z_att_k)
            inter_out_k = intermidiate_outputs[i]
            att_features += [torch.cat([z_att_k, inter_out_k], dim=1)]

        att_features += [torch.nn.UpsamplingBilinear2d(scale_factor=2)(att_features[-1])]

        return att_features

    def forward(self, input_x):
        intermidiate_outputs = self.fw_pass(input_x)
        att_features = self.bw_pass(intermidiate_outputs)

        return att_features
