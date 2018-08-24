# bikeGAN

this is the fork of [BicycleGAN](https://junyanz.github.io/BicycleGAN/) which is used to train and run GANs for [FrankenGAN](http://geometry.cs.ucl.ac.uk/projects/2018/frankengan). The interactive system which uses these networks is [chordatlas](https://github.com/twak/chordatlas)

## running

requirements: 
* nvidia GPU
* pytorch 1.4
* visdom
* dominate

The entry point is `test_interactive.py` which listens to the `./input` folders for new inputs, and writes them to `./output`. Once it is running, set [chordatlas](https://github.com/twak/chordatlas)'s bikeGAN file location (in the settings menu) to the bikeGAN root directory (the one containing this file).

alternatively, use the [container on dockerhub](http://tba).

## cite

if you use this project, please cite [FrankenGAN](https://arxiv.org/abs/1806.07179)

