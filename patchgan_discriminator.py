import torch 
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, num_layers = 4):
        super(PatchGANDiscriminator, self).__init__()

        self.hidden_dim = 64
        self.kernel_size = 4
        self.padding = 1
        self.alpha = 0.2
        self.stride = 2
        prev_multiplier = 1
        multiplier = 1

        model_ = [nn.Conv2d(input_channels, self.hidden_dim, kernel_size= self.kernel_size, stride=self.stride, padding = self.padding),
                  nn.LeakyReLU(self.alpha, True)]
        

        for i in range(1, num_layers):
            prev_multiplier = multiplier
            # Gradually increasing and capping at 8
            multiplier = min(2**i, 8)
            model_.extend([
                nn.Conv2d(prev_multiplier * self.hidden_dim, multiplier * self.hidden_dim, kernel_size = self.kernel_size, stride = self.stride, padding= self.padding),
                nn.BatchNorm2d(multiplier * self.hidden_dim),
                nn.LeakyReLU(self.alpha, True)
            ])

        prev_multiplier = multiplier
        multiplier = min(2**num_layers, 8)

        model_.extend([
            nn.Conv2d(prev_multiplier*self.hidden_dim, multiplier* self.hidden_dim, kernel_size = self.kernel_size, stride=1, padding= self.padding),
            nn.BatchNorm2d(multiplier* self.hidden_dim, True),
            nn.LeakyReLU(self.alpha, True)
        ])

        self.model = nn.Sequential(*model_)

        print("****Discriminator model****")
        print(self.model)
        print("********")

    def forward(self, input_image):
        return self.model(input_image)

if __name__ == "__main__":
    pgd = PatchGANDiscriminator(6)
    input_image = torch.randn(1, 6, 128, 128)
    out = pgd(input_image)
    print(out.shape)

