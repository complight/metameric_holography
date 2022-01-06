# Metameric Varifocal Holography
[David R. Walton](https://drwalton.github.io),
[Koray Kavaklı](https://www.linkedin.com/in/koray-kavakli-75949241/),
[Rafael Kuffner Dos Anjos](https://eps.leeds.ac.uk/computing/staff/9770/dr-rafael-kuffner-dos-anjos),
[David Swapp](http://www0.cs.ucl.ac.uk/people/D.Swapp.html),
[Tim Weyrich](https://reality.cs.ucl.ac.uk/weyrich.html),
[Hakan Ürey](https://mems.ku.edu.tr/),
[Anthony Steed](https://wp.cs.ucl.ac.uk/anthonysteed/),
[Tobias Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/)
and [Kaan Akşit](https://kaanaksit.com)

<img src='https://github.com/complight/metameric_holography/raw/main/teaser.png' width=960>

[Project Page](https://vr.cs.ucl.ac.uk/research/pipelines/metameric-varifocal-holography/) | [Paper](https://arxiv.org/abs/2110.01981) | [Video](https://vimeo.com/623474853)

# Description

This repository contains code related to the paper Metameric Varifocal Holography. It contains scripts to optimise varifocal holograms using metameric loss, or a variety of other possible loss functions. It also contains a blender script which can be used to render virtual images with depth of field with focus on a specified gaze location.

# Installation

This repository makes use of the latest release of `odak`. To install this please use:

`$ pip install odak`

Please ensure that you have installed version `0.1.9` or above.

Depending on your environment, you may need to replace `pip` with `pip3` in the above.

We also make use of [pytorch](https://pytorch.org/get-started/locally/), `opencv2`, `matplotlib` and `numpy`. We recommend installing these with your package manager, `conda` or `pip` as appropriate.

## Hardware requirements

Hologram generation on GPU using the current implementation of the metameric loss requires a significant amount of GPU memory. At 1920x1080 resolution we found that a 2080 Ti with 11GB of video RAM worked well, wheras at 1280x720 a 2070 card with 8GB of RAM was fine. The actual required memory may vary depending on other applications or OS use of the GPU.

# Usage

## Generating holograms

To run the hologram generation code, change your current directory to `code/varifocal_holograms` in this repository, and run `$ python main.py --setting [settings_file]` where `[settings_file]` is an appropriate `JSON`-formatted settings file. 

An example settings file is provided at `code/varifocal_holograms/settings/default.txt`. It will process all images in the provided input directory.

The input directory should contain images and corresponding `JSON` metadata files, which contan information about the images. See `images/magazines` for an example. The only critical piece of information that must be correct in the `JSON` files for hologram generation is the `gaze_location`.

The important settings are as follows:

* `general`
    * `cuda`: 1 to use CUDA, 0 to use CPU.
    * `gpu no`: Index of GPU to use, if CUDA is used.
    * `iterations`: Iteration count to use to optimise each hologram.
    * `propagation type`: Set to "custom" to use the optimised kernel described in the paper. Set to "TR Fresnel" to use an ideal kernel.
    * `output directory`: Place where the holograms will be output to. Directory will be created if it does not already exist.
    * `loss weights`: Controls the weighting of each of the implemented loss functions. See the comment above for information on which is which. Any loss set to 0.0 will not be computed.
    * `region of interest`: Used to set a "do not care" region around the outside of the image, which can improve hologram quality.
    * `learning rate`: Learning rate used by the optimiser (we use ADAM here).
    * `hologram no`: Used for the temporal averaging application. To disable temporal averaging and produce only one hologram per input, set this to 1. For higher numbers, it will generate multiple holograms following the temporal averaging idea described in the paper.
* `metameric`
    * `real viewing distance`: The real distance of the viewer from the image plane. Units are arbitrary, but ensure the same units are used for `real image width`.
    * `real image width`: Real width of the displayed image plane.
* `dataset`
    * `input directory`: Directory containing the images and `JSON` metadata files.
    * `keys`: File extensions for the images and metadata files. Change this e.g. if you use `JPG` images.

## Rendering DoF images

The blender script for rendering images with DoF is located in dataset_rendering. This script was written for blender 2.92 and may not work for earlier blender versions.

To use it, I recommend loading your scene in blender and switching to the "scripting" tab at the top of the screen. From here you can load the script into the text editor built into blender.

Set up the scene as you would like, with the correct camera position. Set the gaze location at the bottom of the script. If desired, you can also use the script to render a grid of focal points or a focus sweep (see the comments in the code for how to do this).

When setting the gaze location, it may be easiest to render a sample image as usual, then use the `dataset_rendering/find_image_point.py` script to select the gaze point you'd like and find its normalised image coordinates. Then copy & paste these into the script.

Render the images by running the script with the "play" button at the top of the text editor.

This will generate an image and the corresponding `JSON` metadata file used by the hologram generation code.

## Further help

For further help with this please contact us via the "issues" section.

