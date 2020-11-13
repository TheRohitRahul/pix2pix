import torch.nn as nn
import torch

from unet_downblock import UnetDownBlock
from unet_upblock import UnetUpBlock

class UnetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, depth = 8, hidden_channels = 64):
        super(UnetGenerator, self).__init__()
        
        total_add = depth - 5

        downblock_model = []

        downblock_model.append( UnetDownBlock(input_channels, hidden_channels, use_batchnorm=True) )
        downblock_model.append( UnetDownBlock(hidden_channels, hidden_channels*2, use_batchnorm=True) )
        downblock_model.append( UnetDownBlock(hidden_channels*2, hidden_channels*4, use_batchnorm=True) )
        downblock_model.append( UnetDownBlock(hidden_channels*4, hidden_channels*8, use_batchnorm=True) )
        downblock_model.append( UnetDownBlock(hidden_channels*8, hidden_channels*8, use_batchnorm=True) )
        
        for i in range(total_add):
            # As the final down block converts feature map to 1x1 we don't want batchnorm there
        
            if total_add - 1 == i:
                downblock_model.append(UnetDownBlock(hidden_channels*8, hidden_channels*8, use_batchnorm=False) )
        
            else:
                downblock_model.append(UnetDownBlock(hidden_channels*8, hidden_channels*8, use_batchnorm=True) )

        upblock_model = []
        dropout_left = 3
        
        for i in range(total_add):
            
            if i == 0:
            
                upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*8, is_innermost=True, add_dropout=True) )
                dropout_left -= 1
            
            else:

                if dropout_left > 0:
                    upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*8, add_dropout=True, is_innermost=False) )
                    dropout_left -= 1

                else:
                    upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*8, add_dropout=False, is_innermost=False) )
        
        if dropout_left > 0:
            upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*8, add_dropout=True, is_innermost=False) )
        else:
            upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*8, add_dropout=False, is_innermost=False) )

        upblock_model.append( UnetUpBlock(hidden_channels*8, hidden_channels*4, add_dropout=False, is_innermost=False) )
        upblock_model.append( UnetUpBlock(hidden_channels*4, hidden_channels*2, add_dropout=False, is_innermost=False) )
        upblock_model.append( UnetUpBlock(hidden_channels*2, hidden_channels, add_dropout=False, is_innermost=False) )
        upblock_model.append( UnetUpBlock(hidden_channels, output_channels, add_dropout=False, is_innermost=False, is_output=True) )
        

        # upblock_model.append( UnetUpBlock(hidden_channels, output_channels, is_output= True, add_dropout=False) )
        self.downblock_model = nn.Sequential(*downblock_model)
        self.upblock_model = nn.Sequential(*upblock_model)

        print("----Generator Model----")
        print("---- downblock ----")
        print(self.downblock_model)
        print("---- upblock ----")
        print(self.upblock_model)
        print("-----------------")

    def forward(self, input_image):

        x = input_image
        down_features = []
        for layer in self.downblock_model:
            x = layer(x)
            down_features.append(x)
        
        down_counter = len(down_features) - 1
        prev_output = down_features[down_counter]

        for up_layer in self.upblock_model:
                
            if up_layer.is_output:
                prev_output = up_layer(prev_output)
                
            else:
                prev_output = up_layer(prev_output, down_features[down_counter - 1 ])
                down_counter -= 1

        return prev_output

if __name__ == "__main__":
    image_size = 128
    image_tensor = torch.randn((1, 3, image_size, image_size))
    unet = UnetGenerator(3, 3, depth=7, hidden_channels=64)
    unet(image_tensor)