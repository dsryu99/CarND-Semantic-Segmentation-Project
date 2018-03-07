# **Semantic Segmentation**
---

## **Semantic Segmentation Project**

The goals / steps of this project are the following:
- In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image References)

[image1]: ./image/loss.jpg "Loss comparison: Epoch 1 vs. Epoch 17"
[image2]: ./image/um_000017.png "An example of labeling the road"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/989/view) individually and describe how I addressed each point in my implementation.  

---
### Build the Neural Network

#### 1. Does the project load the pretrained vgg model?
The function `load_vgg` is implemented correctly. The model and weights are loaded by using `tf.saved_model.loader.load`.

#### 2. Does the project learn the correct features from the images?
The function `layers` is implemented correctly. FCN-8 is correctly implemented. The fully-connected layers are replaced by 1-by-1 convolutions. We upsample the input to the original image size. We also added skip connections to the model.

#### 3. Does the project optimize the neural network?
The function `optimize` is implemented correctly. After the cross entropy loss is calculated, it is minimized by using `tf.train.AdamOptimizer`.

#### 4. Does the project train the neural network?
The function `train_nn` is implemented correctly. The loss of the network should be printed while the network is training. Each image and label is fetched by `get_batches_fn()`. For each iteration of training, the epoch, batch, and loss are printed out.

---
### Neural Network Training
#### 1. Does the project train the model correctly?
On average, the model decreases loss over time. We compared the loss of epoch 1 with that of epoch 17 as follows:
![alt text][image1]

#### 2. Does the project use reasonable hyperparameters?
The number of epoch and batch size are set to a reasonable number. The number of epoch and batch isze are set to 30 and 2 respectively.

#### 3. Does the project correctly label the road?
The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. My solution labels at least 80% of the road and label no more than 20% of non-road pixels as road. The following image is the example of labeling the road.
![alt text][image2]
