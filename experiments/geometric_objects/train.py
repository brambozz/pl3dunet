import pytorch_lightning as pl
import pl3dunet.unet as unet
import generate_dataloader

# Define train dataloader
def train_dataloader():
    return generate_dataloader.get_dataloader()


# Initialize network
model = unet.UNet(in_channels=1, out_channels=5)
model.train_dataloader = train_dataloader

trainer = pl.Trainer()
trainer.fit(model)
