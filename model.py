import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)

class UpConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)

class CongestionModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Encoder
        self.conv1 = ConvBlock(dim_in, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        # Decoder
        self.dec_conv1 = ConvBlock(32, 32)
        self.upconv1 = UpConvBlock(32, 16)
        self.dec_conv2 = ConvBlock(16, 16)
        self.upconv2 = UpConvBlock(32+16, 4)  # +32 for skip connection
        self.final = nn.Sequential(
            nn.Conv2d(4, dim_out, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        
        # Decoder
        dec1 = self.dec_conv1(conv3)
        up1 = self.upconv1(dec1)
        dec2 = self.dec_conv2(up1)
        # Concatenate skip connection from encoder
        up2 = self.upconv2(torch.cat([dec2, pool1], dim=1))
        output = self.final(up2)
        
        return output

if __name__ == '__main__':
    from congestion_dataset import CongestionDataset
    
    dataset = CongestionDataset()
    input_data, target_data = dataset[0]
    print("Input data shape:", input_data.shape)
    print("Target data shape:", target_data.shape)
    
    model = CongestionModel(dim_in=input_data.shape[0], dim_out=target_data.shape[0])
    output = model(input_data.unsqueeze(0))
    print("Model output shape:", output.shape)
    print(model)