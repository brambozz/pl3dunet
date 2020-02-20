import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_3d(sample):
    """Show a 3d sample image

    Parameters
    ----------
    sample
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.voxels(sample)
    plt.show()


def show_sample(sample):
    """Show a 3d sample image

    Parameters
    ----------
    sample : dict
        sample["image"] : array with shape (size, size, size)
            The image with num_obj objects randomly placed
        sample["masks"] : array
            Same size as image, but with a channel per object type.
            Each channel is the segmentation mask for each object type.
    """
    # Combine masks into single image with color array
    masks = np.zeros_like(sample["image"])
    masks_color = np.zeros(sample["image"].shape, dtype=object)
    color_list = ["red", "blue", "green", "orange", "purple"]
    for i in range(5):
        masks = masks | sample["masks"][i]
        masks_color[sample["masks"][i]] = color_list[i]

    # Prepare figure
    fig = plt.figure()
    ax_image = fig.add_subplot(1, 2, 1, projection="3d")
    ax_masks = fig.add_subplot(1, 2, 2, projection="3d")

    # Plot image and masks
    ax_image.voxels(sample["image"])
    ax_masks.voxels(masks, facecolors=masks_color)

    plt.show()
