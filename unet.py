"""Lightning module for a 3D u-net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def create_double_conv(channels):
    """Create a double convolution+BN+ReLU block

    Parameters
    ----------
    channels : list or tuple
        Holds channels of input, middle and output layer of encoder block

    Returns
    -------
    DoubleConv : nn.Sequential
    """
    return nn.Sequential(
        nn.Conv3d(channels[0], channels[1], 3),
        nn.BatchNorm3d(channels[1]),
        nn.ReLU(),
        nn.Conv3d(channels[1], channels[2], 3),
        nn.BatchNorm3d(channels[2]),
        nn.ReLU(),
    )


class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define encoder convolution blocks
        self.encoder_conv1 = create_double_conv([in_channels, 32, 64])
        self.encoder_conv2 = create_double_conv([64, 64, 128])
        self.encoder_conv3 = create_double_conv([128, 128, 256])
        self.encoder_conv4 = create_double_conv([256, 256, 512])

        # Define decoder convolution blocks
        # self.decoder_conv1 = create_double_conv([256 + 512, 256, 256])
        # self.decoder_conv2 = create_double_conv([128 + 256, 128, 128])
        # self.decoder_conv3 = create_double_conv([64 + 128, 64, 64])
        self.decoder_conv1 = create_double_conv([512, 256, 256])
        self.decoder_conv2 = create_double_conv([256, 128, 128])
        self.decoder_conv3 = create_double_conv([128, 64, 64])

        # Define maxpool
        self.maxpool = nn.MaxPool3d(2)  # Kernel size and stride 2

        # Define upconv
        self.upconv1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        # Define final convolution
        self.outconv = nn.Conv3d(64, self.out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x = self.maxpool(x1)
        x2 = self.encoder_conv2(x)
        x = self.maxpool(x2)
        x3 = self.encoder_conv3(x)
        x = self.maxpool(x3)
        x = self.encoder_conv4(x)

        # Decoder
        x = self.upconv1(x)
        # x = torch.cat((x3, x), dim=1)
        # TODO: center crop with correct dimensions
        x = self.decoder_conv1(x)
        x = self.upconv2(x)
        # x = torch.cat((x2, x), dim=1)
        x = self.decoder_conv2(x)
        x = self.upconv3(x)
        # x = torch.cat((x1, x), dim=1)
        x = self.decoder_conv3(x)

        # Final convolution
        x = self.outconv(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.99)

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": val_loss_mean}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {"test_loss": F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": test_loss_mean}
