import torch.nn as nn 


class Pix2Pix(nn.Module):
    def __init__(self, in_channel = 3):
        super().__init__()

        self.upsample_pix = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 2048, kernel_size=(1,1)),
        )

        self.upsample_pos = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 2048, kernel_size=(1,1)),
        )

        self.downsample_comb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2048, 32, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1,1)),
        )

    def forward(self, image, pos_map):
        out = self.upsample_pix(image) + self.upsample_pos(pos_map)
        return self.downsample_comb(out)
