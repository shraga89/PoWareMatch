# *PoWareMatch*: Matching as a Process and the Case of a Human Matcher
<p align="center">
<img src ="/fig.JPG">
</p>

## Prerequisites:
1. [Anaconda 3](https://www.anaconda.com/download/)
2. [Pytorch](https://pytorch.org/)


## Getting Started

### Installation:
1. Get human (algorithmic) matching data.
1.1. An example dataset is available for download: [Human Dataset Sample]()
1.2. An example reference match is given in [Reference Match Example]()
1.3. An example algorithmic match is given in [Algorithmic Match Example]() 
2. Clone the [PoWareMatch repository](https://github.com/shraga89/PoWareMatch)
3. Update [Config](https://github.com/shraga89/PoWareMatch/blob/master/Config.py) with your configuration details.

### Usage
1. [RunFiles]() folder consists of several runnables to train and test your dataset using a 5-fold cross validation.
1.1 The runnable files differ in the extent of which the experiments are run (e.g., use [pred_y]() to only train and test a calibration model)  
2. Once the models are trained, you can use the [PoWareMatch inference]() to calibrate human matching and generate better matches.
 
## The Paper
The paper is under submission.

## The Team
*ADnEV* was developed at the Technion - Israel Institute of Technology by [Roee Shraga](https://sites.google.com/view/roee-shraga/) under the supervision of [Prof. Avigdor Gal](https://agp.iem.technion.ac.il/avigal/)