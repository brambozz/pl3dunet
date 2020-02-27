import torch

# Generate as follows

# Load single image into memory

# Apply mirroring strategy, find some premade code for this

# Use 'unfold' to break up the image into non-overlapping patches
x = torch.randn(500, 500, 496)
x = x.unfold(0, 132, 132)
x = x.unfold(1, 132, 132)
x = x.unfold(2, 116, 116)
x = x.contiguous().view(-1, 132, 132, 116)

# Simililarly for the masks

# Stack all patches in the batch dimension and return item
