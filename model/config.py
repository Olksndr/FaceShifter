
class Config:
    AADConfig = {
    0:{'input_dim': 1024,
      'output_dim': 1024,
      'z_input_channels': [512, 1024],
      'AADUnitConv_params':{'kernel_size': [2, 2], 'stride': 1, 'padding': 0}
          },
    1:{'input_dim': 1024,
      'output_dim': 1024,
      'z_input_channels': [512, 2048],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    2:{'input_dim': 1024,
      'output_dim': 1024,
      'z_input_channels': [512, 1024],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    3:{'input_dim': 1024,
      'output_dim': 512,
      'z_input_channels': [512, 512],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    4:{'input_dim': 512,
      'output_dim': 256,
      'z_input_channels': [512, 256],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    5:{'input_dim': 256,
      'output_dim': 128,
      'z_input_channels': [512, 128],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    6:{'input_dim': 128,
      'output_dim': 64,
      'z_input_channels': [512, 64],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    7:{'input_dim': 64,
      'output_dim': 3,
      'z_input_channels': [512, 64],
      'AADUnitConv_params':{'kernel_size': [3, 3], 'stride': 1, 'padding': 1}
          },
    }

    MLAttConfig = {'input_channels': [3, 32, 64, 128, 256, 512, 1024],
                    'output_channels': [32, 64, 128, 256, 512, 1024, 1024]
                            }


    # _opt = namedtuple('')
