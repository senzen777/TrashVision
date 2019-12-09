# TrashVision
Trash sorting and labeling computer vision system

## Source Code

We did most of our software development over Google Colab. Our notebooks are available in the `notebooks` folder:

- `OrganicNet.ipynb` is our code for training the CNN based on the Kaggle dataset
- `texturerecognitionFinalTwo.ipynb` is our code for training the CNN based on the TrashNet dataset
- `Together.ipynb` is the code for integrating our two CNNs together
- `FullNet.ipynb` is the code to train our single-CNN point of comparison
- `blob_detect.py` is the code we used to try blob detection to preprocess our images (we don't use it in the final product since it didn't improve accuracy)

## Demonstration

You can run our method on sample images using `TrashVision.py`

Run:

`python3 TrashVision.py`

We've tested this on python 3.7.0 with PyTorch.

To run on our sample images, enter `images/` when prompted. Sample images starting with `c` are to be composted, starting with `r` are to be recycled, starting with `t` are to be incinerated/trashed.

We have very high accuracy on `r` and `c` images, but we get very low accuracy on `t` images due to a lack of training images.

Feel free to also test on other `.jpg` images if you wish.

## Data Sources
TrashNet Data: https://github.com/garythung/trashnet

Kaggle Data: https://www.kaggle.com/techsash/waste-classification-data
