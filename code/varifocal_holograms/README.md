# Metameric Varifocal Hologram
This codebase introduces an optimization to identify phase-only holograms to reconstruct images that are metamerized. 
The technical details of the work are detailed in our [manuscript](). 
We will skip describing the work and help you to get this codebase running at your end.

## Quickstart
### (0) Install the required packages
** DISCLAIMER: If you want this codebase to run with GPU support, please make sure to install `pytorch` with GPU support before installing odak.**

To run this codebase, you have to install the required packages.
Run `pip3 install odak`.
That should hopefully get you going with your setup.
At the time of writing, odak is at `0.1.9` and requires Python `3.7` or above.

Please note that version `0.1.9` or above of odak is required to run this code.

### (1) Running the code to optimize holograms with default settings.
To run an optimization with default settings and targets, simply run:
```
python3 main.py
```
within this directory. This will immediatly read the dataset readily available in [here](../../images/depth_of_field).
If you want to work with metameric loss, visit the [settings file](settings/default.txt) and set the loss weights in the general group to 1.0,0.0.
If you set the same value to 0.0 and 1.0, that will use L2 loss in optimization rather than metameric loss.
The results of the optimization will always end up at a new directory named as `output`.
This directory will automatically be created during runtime.

### (2) Running the code for some other targets.
If you are willing to run the optimizations for another set of images, please visit the [settings file](settings/default.txt). 
As you visit this file, you must change the set value for the dataset's input directory.
If you are curious about how the dataset is generated in the first place so that you can generate your dataset that fits nicely with this code, please visit [dataset rendering code folder](../dataset_rendering/).

### (3) What if I have questions?
Different parties prepare different parts of this code.
The dataset generation and metameric loss function parts are prepared by [David Walton](https://drwalton.github.io).
The hologram optimization part is prepared by [Kaan Akşit](https://kaanaksit.com).
[Rafael Kuffner dos Anjos](https://rafaelkuffner.github.io/) verified that this README works for both on Windows and Linux with pip or conda.
Please do not hesitate to use issues section or reach out to these people directly.
