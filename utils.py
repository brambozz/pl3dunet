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
    # TODO: each mask a separate color
    # Prepare figure
    fig = plt.figure()
    ax_list = []
    for i in range(6):
        ax_list.append(fig.add_subplot(2, 3, i + 1, projection="3d"))

    # Plot image
    ax_list[0].voxels(sample["image"])

    # Plot masks
    for i in range(5):
        ax_list[i + 1].voxels(sample["masks"][i])

    plt.show()
