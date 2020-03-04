# PyTorch Lightning 3D u-net

This is a 
[PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning#examples)
implementation of a 3D u-net.
This implementation tries to be as faithful as possible to the original
[publication](https://arxiv.org/abs/1606.06650) and is therefore 
barebones, i.e. without a lot of configuration options. 

For a more extensive implementation, see the lightning implementation of
[pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). 
Will I do this in the future???

## Usage

* Use this repository as a git submodule in your project

## Dependencies

[imgaug](https://github.com/aleju/imgaug).

## Example

This will describe a toy example with synthetic 3D data. I'm thinking
of something with various geometric object. Perhaps I can find something
elsewhere, I remember some presentation about learning 3D segmentation
from a 2D network.

## References

The toy data generation code was inspired from 
[ACSConv](https://github.com/M3DV/ACSConv).

The model code was inspired from
[pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
