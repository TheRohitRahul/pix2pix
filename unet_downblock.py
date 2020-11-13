import torch.nn as nn

class UnetDownBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, use_dropout = False, use_batchnorm = False):
        super(UnetDownBlock, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size = 4, stride=2, padding = 1)
        self.activation = nn.LeakyReLU(0.2)
        
        if use_batchnorm:
            self.batch_norm = nn.BatchNorm2d(hidden_channels)
        
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
    def forward(self, input_feature_map):
        output_conv = self.conv(input_feature_map)
        if (self.use_batchnorm):
            output_conv = self.batch_norm(output_conv)
        output_conv = self.activation(output_conv)
        return output_conv


