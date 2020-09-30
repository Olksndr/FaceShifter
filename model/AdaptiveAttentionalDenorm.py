import torch



class AADUnit(torch.nn.Module):

    def __init__(self, h_input_channels, z_input_channels, output_channels,
                    conv_params):
        super(AADUnit, self).__init__()
        z_id_input_ch, z_att_input_ch = z_input_channels
        self.batch_norm = torch.nn.BatchNorm2d(output_channels)
        #adaptation mask M
        self.conv_mask = torch.nn.Conv2d(h_input_channels, output_channels,
            conv_params['kernel_size'], stride=conv_params['stride'],
                padding=conv_params['padding'])
        #identity integration
        self.gamma_fc = torch.nn.Linear(z_id_input_ch, output_channels)
        self.beta_fc = torch.nn.Linear(z_id_input_ch, output_channels)
        #attributes integration
        self.gamma_conv = torch.nn.Conv2d(z_att_input_ch, output_channels,
            kernel_size=conv_params['kernel_size'], stride=conv_params['stride'],
                padding=conv_params['padding'])
        self.beta_conv = torch.nn.Conv2d(z_att_input_ch, output_channels,
            kernel_size=conv_params['kernel_size'], stride=conv_params['stride'],
                padding=conv_params['padding'])


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
    def __init__(self, input_channels, output_channels, z_input_channels, conv_params):
        super(DefaultResBLK, self).__init__()
        self.AAD = AADUnit(h_input_channels=input_channels,
                z_input_channels=z_input_channels,
                output_channels=input_channels,
                conv_params=conv_params ####
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
    def __init__(self, input_channels, output_channels, z_input_channels, conv_params):
        super(AADResBLK, self).__init__()
        self.additional_block = True if input_channels != output_channels else False

        if self.additional_block:

            self.res_blk_0 = DefaultResBLK(input_channels=input_channels, output_channels=input_channels,
                                                           z_input_channels=z_input_channels, conv_params=conv_params)
            self.res_blk_1 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels, conv_params=conv_params)
            self.res_blk_2 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels, conv_params=conv_params)
        else:
            self.res_blk_0 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels, conv_params=conv_params)
            self.res_blk_1 = DefaultResBLK(input_channels=input_channels, output_channels=output_channels,
                                                           z_input_channels=z_input_channels, conv_params=conv_params)

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
    def __init__(self, aad_config):
        super(AADGenerator, self).__init__()
        self.layers = []
        self.h_init_layer = torch.nn.ConvTranspose2d(512, 1024, [2, 2], 1, 0)
        for params in aad_config.values():
            layer = AADResBLK(params['input_dim'], params['output_dim'],
                params['z_input_channels'], params['AADUnitConv_params'])
            self.layers += [layer]
    def forward(self, inputs):
        z_attributes_dict, z_id = inputs
        h_init = self.h_init_layer(z_id)
        #hidden_init, z_identity, z_attributes = inputs
        hidden = h_init
        for k in range(8):
            resblk_inputs = [hidden, z_id, z_attributes_dict[k]]
            # print(k)
            hidden = self.layers[k](resblk_inputs)
            # print(hidden.shape , "before upsampling")
            if k == 7:
                return hidden
            hidden = torch.nn.UpsamplingBilinear2d(scale_factor=2)(hidden) ###!
            # print(hidden.shape , "After upsampling")
        # return hidden
