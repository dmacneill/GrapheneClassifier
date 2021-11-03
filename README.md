# GrapheneClassifier
Since 2004 [[1]](#1) two-dimensional materials have been one of the most active areas of research in materials science. The most common method to prepare two-dimensional materials is mechanical exfolitation: a non-deterministic procedure where we press a thick crystal onto a silicon wafer, leaving behind crystal fragments which are occasionally only a few atoms thick. For example, to prepare graphene (a monolayer of carbon atoms) we press graphite crystals onto a silicon wafer. We use a microscope to search the debris for thin crystals, which are typically about 4-10% darker than the bare wafer. Some images of crystallites prepared by mechanical exfoliation are shown below:

<p align = "center">
<img src="figures/fig1.png" width=900>
</p>
  
*Left: high-magnification image of exfoliated monolayer graphene. Right: low-magnification image of the same graphene monolayer along with graphite crystals of various
thicknesses (up to 100s of nanometers).*

Depending on the material this search (colloquially termed "flake-hunting") can be quite time-consuimg, sometimes taking several hours. Many research groups have tried to automate the process, but so far no solution has been convenient or reliable enough for widespread adoption. The group I work in at MIT has also put efforts into automated flake-hunting [[2]](#2), and this project is a continuation of that. 

Previous work showed convolutional neural nets (CNN) can be trained to identify thin crystals, but those models were trained on high-magnification images which limits throughput (since it takes a long time to image an entire wafer at high-magnification). Here, I discuss some experiments on flake-identification using an objective lens with low-magnification and large field-of-view (each image captures a 2.8 mm by 1.9 mm region). Another barrier preventing automated flake hunting is that different research projects have different materials requirements. Researchers might want to quickly characterize a material that has never been exfoliated before, or to find crystals with a specific shape. To this end I wanted to see if self-supervised pretraining with contrastive learning [[3]](#3) could increase data efficieny, and will compare the results with models trained from scratch.

### Requirements
Python 3.8.10, Pillow 7.2.0, NumPy 1.19.1, Matplotlib 3.3.4, PyTorch 1.8.1

### Usage

The main module is `train.py` which uses gradient descent to train the model defined by the class `Classifier` in `model.py`. It is designed to be run from the command line. To see the arguments, call `python train.py -h`. Arguments can be passed from file usin `@` as a prefix. The module `clr.py` can be run from the command line to carry out contrastive learning, again using the `Classifier` class defined in `model.py`. 

As described in [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf), for contrastive learning we train the model with a projection head that outputs a high-dimensional embedding. To fine-tune on the actual data we replace this projection head with a linear map that outputs a `num_classes` dimensional vector. This is handled automatically by the `Classifier` class, which can be initialized in either configuration. During training `clr.py` saves weights for the entire model and also weights for the backbone only. To use `train.py` with a pre-trained backbone use the command line argument `--backbone_weights_path`. 

Finally, the model defined in `model.py` is a small CNN with four convolutional layers that is not really appropriate for contrastive learning. `resnet.py` defines a classifier based on ResNet-50; to use it rename this file to `model.py`.

### Results

To train the model to identify graphene I created a dataset of 3756 training images and 683 validation images. The images are 400X400 pixel squares cropped from 4908X3264 microscope images covering a 2.8 mm by 1.9 mm field of view. The dataset contains: i) selected images of thin graphene/graphite films, ii) selected images of thick graphite crystals and other non-graphene objects on the wafer (residues, dirt, etc.), and iii) randomly cropped regions. For each image, I created labels indicating the non-exclusive presence or absence of three classes: 
Class 1: Graphene films with R channel contrast <= 12% 
Class 2: Graphene films with R channel contrast >12% and <=19%
Class 3: Graphene films with R channel contrast >19% and <=30%

### References
<a id="1">[1]</a> K.S. Novoselov *et al.,* "Electric field effect in atomically thin carbon films", *Science* **306**, 666-669 (2004)

<a id="2">[2]</a> B. Han *et al.,* "Deep-Learning-Enabled Fast Optical Identification and Characterization of 2D Materials", *Advanced Materials* **32**, 2000953 (2020)

<a id="3">[3]</a> T. Chen *et al.,* "Big Self-Supervised Models are Strong Semi-Supervised Learners", arXiv preprint arXiv:2006.10029 (2020)
