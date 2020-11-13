import torch.nn as nn
import torch

class UnetUpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, is_output = False, add_dropout = False, is_innermost = False):
        super(UnetUpBlock, self).__init__()
        # Inner most won't get double the channels as after this deconv we will start concatenating channels
        if not is_innermost:
            self.deconv = nn.ConvTranspose2d(input_channels*2, output_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)

        
        if not is_output:
            self.batch_norm = nn.BatchNorm2d(output_channels)
        
        self.activation = nn.ReLU()
        if is_output:
            self.activation = nn.Tanh()

        if add_dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.add_dropout = add_dropout
        self.is_output = is_output

    def forward(self, input_feature_map, skip_connection = None):
        out_deconv = self.deconv(input_feature_map)
        if not self.is_output:
            out_deconv = self.batch_norm(out_deconv)

        out_deconv = self.activation(out_deconv)

        if self.add_dropout:
            out_deconv = self.dropout(out_deconv)
        
        if not self.is_output:
            out_deconv = torch.cat([out_deconv, skip_connection], dim=1)

        return out_deconv