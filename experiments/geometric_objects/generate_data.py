import numpy as np

# TODO: optional rotation of objects


def pyramid(base, height):
    x, y, z = np.indices((base, base, height))
    r = np.linspace(0, base / 2, height)
    r = np.tile(r.reshape(1, 1, height), (base, base, 1))
    obj = np.maximum(np.abs(x - (base - 1) / 2), np.abs(y - (base - 1) / 2)) <= r
    return obj


def cone(diameter, height):
    x, y, z = np.indices((diameter, diameter, height))
    r = np.linspace(0, diameter / 2, height)
    r = np.tile(r.reshape(1, 1, height), (diameter, diameter, 1))
    obj = np.sqrt((x - (diameter - 1) / 2) ** 2 + (y - (diameter - 1) / 2) ** 2) <= r
    return obj


def ball(diameter, _):
    x, y, z = np.indices((diameter, diameter, diameter))
    obj = (
        np.sqrt(
            (x - (diameter - 1) / 2) ** 2
            + (y - (diameter - 1) / 2) ** 2
            + (z - (diameter - 1) / 2) ** 2
        )
        <= diameter / 2
    )
    return obj


def cube(base, _):
    return np.ones((base, base, base), dtype=bool)


def cylinder(diameter, height):
    x, y, z = np.indices((diameter, diameter, height))
    r = np.array([diameter / 2,])
    r = np.tile(r.reshape(1, 1, 1), (diameter, diameter, height))
    obj = np.sqrt((x - (diameter - 1) / 2) ** 2 + (y - (diameter - 1) / 2) ** 2) <= r
    return obj


def generate_sample(num_instance, size, num_obj=4):
    """Generate a training sample

    Parameters
    ----------
    num_instance : int
        Amount of training samples to generate
    size : int
        Dimension of cubic image
    num_obj : int
        Number of objects to place
        
    Returns
    -------
    sample : dict
        sample["image"] : array with shape (size, size, size)
            The image with num_obj objects randomly placed
        sample["masks"] : array
            Same size as image, but with a channel per object type.
            Each channel is the segmentation mask for each object type.
    """
    # Define object functions
    obj_funcs = [cone, pyramid, cube, cylinder, ball]

    # Initialize empty sample
    sample = {}
    sample["image"] = np.zeros((size, size, size), dtype=bool)
    sample["masks"] = np.zeros((len(obj_funcs), size, size, size), dtype=bool)

    # Place objects iteratively
    for i in range(num_obj):
        # Determine object dimensions randomly
        dim1, dim2 = np.random.randint(5, size // 4, size=2)

        # Determine place (corner of object) randomly
        place = np.random.randint(0, size - np.max([dim1, dim2]), size=3)

        # Generate random object
        obj_func = np.random.randint(len(obj_funcs))
        obj = obj_funcs[obj_func](dim1, dim2)

        # Place object in empty image
        dim1, dim2, dim3 = obj.shape
        obj_image = np.zeros_like(sample["image"])
        obj_image[
            place[0] : place[0] + dim1,
            place[1] : place[1] + dim2,
            place[2] : place[2] + dim3,
        ] = obj

        # Place object in appropriate mask
        obj_mask = np.zeros_like(sample["masks"])
        obj_mask[
            obj_func,
            place[0] : place[0] + dim1,
            place[1] : place[1] + dim2,
            place[2] : place[2] + dim3,
        ] = obj

        # Image is union
        sample["image"] = sample["image"] | obj_image
        sample["masks"] = sample["masks"] | obj_mask

    return sample
