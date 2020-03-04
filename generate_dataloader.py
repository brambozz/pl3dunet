from torch.utils.data import Dataset, DataLoader
import numpy as np
import generate_data


class GeometricObjectsDataset(Dataset):
    """Dataset of 3D images containing random geometric objects"""

    def __init__(self, epoch_size=100, image_size=28, num_obj=4, input_size=116):
        """
        Parameters
        ----------
        epoch_size : int
            Size of one epoch
        image_size : int
            Size of the image
        input_size : int
            Size of input to the network
        num_obj : int
            Number of objects in each image
        """
        self.epoch_size = epoch_size
        self.image_size = image_size
        self.input_size = input_size
        self.num_obj = num_obj

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Generate random sample
        sample = generate_data.generate_sample(1, self.image_size, self.num_obj)
        image, masks = (
            sample["image"].astype(np.float32),
            sample["masks"].astype(np.float32),
        )

        # Reflection pad image and masks to input size
        padding = (self.input_size - self.image_size) // 2
        input = np.pad(image, padding, mode="reflect")

        # Add channel dimension to image and input
        image = image[None, :, :, :]
        input = input[None, :, :, :]

        return {"image": image, "input": input, "masks": masks}


def get_dataloader(
    batch_size=1, epoch_size=100, image_size=28, num_obj=4, input_size=116
):
    """Create dataloader for geometric objects dataset

    Parameters
    ----------
    batch_size : int
    epoch_size : int
        Size of one epoch
    image_size : int
        Size of the image
    input_size : int
        Size of input to the network
    num_obj : int
        Number of objects in each image

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
    """
    dataset = GeometricObjectsDataset(input_size)
    return DataLoader(dataset, batch_size=batch_size)
