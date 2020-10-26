import torch

class AADUnit(torch.nn.Module):

    def __init__(self, h_input_channels, z_input_channels, output_channels,
                    kernel_size=1, stride=1, padding=0):
        super(AADUnit, self).__init__()
        z_id_input_ch, z_att_input_ch = z_input_channels
        self.batch_norm = torch.nn.BatchNorm2d(output_channels)
        #adaptation mask M
        self.conv_mask = torch.nn.Conv2d(h_input_channels, output_channels,
            kernel_size, stride=stride,
                padding=padding)

        #identity integration
        self.gamma_fc = torch.nn.Linear(z_id_input_ch, output_channels)
        self.beta_fc = torch.nn.Linear(z_id_input_ch, output_channels)

        #attributes integration
        self.gamma_conv = torch.nn.Conv2d(z_att_input_ch, output_channels,
            kernel_size=kernel_size, stride=stride,
                padding=padding)
        self.beta_conv = torch.nn.Conv2d(z_att_input_ch, output_channels,
            kernel_size=kernel_size, stride=stride,
                padding=padding)


    def forward(self, inputs):
        hidden, z_identity, z_attributes = inputs

        hidden_norm = self.batch_norm(hidden)

        #adaptation mask
        M = self.conv_mask(hidden_norm)
        M = torch.nn.Sigmoid()(M)

        #identity integration
        gamma_id = self.gamma_fc(z_identity.squeeze(-1).squeeze(-1))
        beta_id = self.beta_fc(z_identity.squeeze(-1).squeeze(-1))

        #attributes integration
        gamma_att = self.gamma_conv(z_attributes)
        beta_att = self.beta_conv(z_attributes)

        "(5) https://arxiv.org/pdf/1912.13457.pdf"
        I_mul = torch.mul(gamma_id.unsqueeze(-1).unsqueeze(-1), hidden_norm)
        I = I_mul + beta_id.unsqueeze(-1).unsqueeze(-1)

        A_mul = torch.mul(gamma_att, hidden_norm)
        A = A_mul + beta_att

        next_h = torch.mul((1 - M), A) + torch.mul(M, I)
        return next_h

class DefaultResBLK(torch.nn.Module):
    def __init__(self, input_channels, output_channels, z_input_channels):
        super(DefaultResBLK, self).__init__()
        self.AAD = AADUnit(h_input_channels=input_channels,
                z_input_channels=z_input_channels,
                output_channels=input_channels
                       )
        self.conv = torch.nn.Conv2d(input_channels, output_channels, [3, 3], 1, 1)
        self.relu_op = torch.nn.ReLU()
    def forward(self, inputs):
        hidden, z_identity, z_attributes = inputs
        hidden = self.AAD([hidden, z_identity, z_attributes])
        hidden_r = self.relu_op(hidden)
        next_hidden = self.conv(hidden_r)

        return next_hidden

class AADResBLK(torch.nn.Module):
    def __init__(self, input_channels, output_channels, z_input_channels):
        super(AADResBLK, self).__init__()
        self.additional_block = True if input_channels != output_channels else False

        if self.additional_block:

            self.res_blk_0 = DefaultResBLK(input_channels=input_channels, output_channels=input_channels,
                                                           z_input_channels=z_input_channels)
            self.res_blk_1 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels)
            self.res_blk_2 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels)
        else:
            self.res_blk_0 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels)
            self.res_blk_1 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels)

    def forward(self, inputs):
        hidden_init, z_identity, z_attributes = inputs
        if self.additional_block:

            hidden_0 = self.res_blk_0([hidden_init, z_identity, z_attributes])
            hidden_1 = self.res_blk_1([hidden_0, z_identity, z_attributes])

            hidden_2 = self.res_blk_2([hidden_init, z_identity, z_attributes])

            return hidden_1 + hidden_2
        else:
            hidden_0 = self.res_blk_0([hidden_init, z_identity, z_attributes])
            hidden_1 = self.res_blk_1([hidden_0, z_identity, z_attributes])
            return hidden_init + hidden_1

class AADGenerator(torch.nn.Module):
    def __init__(self):
        super(AADGenerator, self).__init__()

        self.h_init_layer = torch.nn.ConvTranspose2d(512, 1024, [2, 2], 1, 0)

        self.res_block_1 = AADResBLK(1024, 1024, [512, 1024])
        self.res_block_2 = AADResBLK(1024, 1024, [512, 2048])
        self.res_block_3 = AADResBLK(1024, 1024, [512, 1024])
        self.res_block_4 = AADResBLK(1024, 512, [512, 512])
        self.res_block_5 = AADResBLK(512, 256, [512, 256])
        self.res_block_6 = AADResBLK(256, 128, [512, 128])
        self.res_block_7 = AADResBLK(128, 64, [512, 64])
        self.res_block_8 = AADResBLK(64, 3, [512, 64])


    def forward(self, inputs):
        [z_1_att, z_2_att, z_3_att, z_4_att, z_5_att, z_6_att,
        z_7_att, z_8_att], z_id = inputs

        hidden_0 = self.h_init_layer(z_id)

        hidden_1 = self.res_block_1([hidden_0, z_id, z_1_att])
        hidden_1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_1)

        hidden_2 = self.res_block_2([hidden_1, z_id, z_2_att])
        hidden_2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_2)

        hidden_3 = self.res_block_3([hidden_2, z_id, z_3_att])
        hidden_3 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_3)

        hidden_4 = self.res_block_4([hidden_3, z_id, z_4_att])
        hidden_4 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_4)

        hidden_5 = self.res_block_5([hidden_4, z_id, z_5_att])
        hidden_5 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_5)

        hidden_6 = self.res_block_6([hidden_5, z_id, z_6_att])
        hidden_6 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_6)

        hidden_7 = self.res_block_7([hidden_6, z_id, z_7_att])
        hidden_7 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden_7)

        hidden_8 = self.res_block_8([hidden_7, z_id, z_8_att])
        y = hidden_8

        return y
