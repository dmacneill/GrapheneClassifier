# GrapheneClassifier
Since 2004 [[1]](#1) layered crystals (called two-dimensional or van der Waals materials) have been one of the most active areas of research in materials science. The most common method to prepare two-dimensional materials is mechanical exfolitation: a non-deterministic procedure where we press a thick crystal onto a silicon wafer, leaving behind crystal fragments which are occasionally only a few atoms thick. For example, to prepare graphene (a monolayer of carbon atoms) we press graphite crystals onto a silicon wafer. We use a microscope to search the debris for thin crystals, which are typically about 4-10% darker than the bare wafer. Some images of crystallites prepared by mechanical exfoliation are shown below:

<p align = "center">
<img src="figures/exfoliation_examples.png" width=1000>
</p>
  
*Left: high-magnification image of exfoliated monolayer graphene. Right: low-magnification image of graphite crystals with varying thicknesses (from monolayer to 100s of nanometers) prepared by mechanical exfoliation.*

Depending on the material this search (colloquially termed "flake-hunting") can be quite time-consuimg, sometimes taking several hours. Many research groups have tried to automate the process, but so far no solution has been convenient or reliable enough for widespread adoption. The group I work in at MIT has also put efforts into automated flake-hunting [[2]](#2), and this project is a continuation of that. 

Previous work showed convolutional neural nets (CNN) can be trained to identify thin crystals very well, but those models were trained on high-magnification images which limits throughput (since microscopes take a long time to image an entire wafer at high-magnification). Here, I discuss some experiments on flake-identification using an objective lens with low-magnification and a large field-of-view. Another barrier preventing automated flake hunting is that different research projects have different materials requirements. Researchers might want to quickly characterize a material that has never been exfoliated before, or to find crystals with a specific shape. To this end I wanted to see if self-supervised pretraining with contrastive learning [[3]](#3) could increase data efficieny, and will compare the results with models trained from scratch.

### Requirements

### Usage

### Results

### References
<a id="1">[1]</a> K.S. Novoselov *et al.,* "Electric field effect in atomically thin carbon films", *Science* **306**, 666-669 (2004)

<a id="2">[2]</a> B. Han *et al.,* "Deep-Learning-Enabled Fast Optical Identification and Characterization of 2D Materials", *Advanced Materials* **32**, 2000953 (2020)

<a id="3">[3]</a> T. Chen *et al.,* "Big Self-Supervised Models are Strong Semi-Supervised Learners", arXiv preprint arXiv:2006.10029 (2020)
