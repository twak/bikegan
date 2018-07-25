# bikeGAN

this is the fork of [BicycleGAN](https://junyanz.github.io/BicycleGAN/) which is used to train and run GANs for [FrankenGAN](http://geometry.cs.ucl.ac.uk/projects/2018/frankengan).

## running

requirements: 
* nvidia GPU
* pytorch 1.4
* visdom
* dominate

`test_interactive.py`, listens to the `./input` folders for new inputs, and writes them to `./output`. Once it is running, set [chordatlas](https://github.com/twak/chordatlas)' bikeGAN file location (in the settings menu) to the bikeGAN root directory (the one containing this file).

alternatively, use the container on dockerhub.

## cite

if you use this project, please cite [FrankenGAN](https://arxiv.org/abs/1806.07179)

