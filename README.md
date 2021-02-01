# CSE 6363-003 Machine Learning

# Term Project final report

# Human Counting on a Frame using OpenCV

## Neel Suthar Manthankumar Patel Jugal Patel

```
neel.suthar@mavs.uta.edu manthankumarlax.patel@mavs.uta.edu jugal.patel@mavs.uta.edu
```
## Abstract

Object detection is a computer technology related to computer
vision and image processing. It allows us to identify and
locate objects in an image or video (we can also use it on live
stream). Object detection can be used to count objects in a
scene and determine and track their precise locations, all while
accurately labeling them which is very useful for many
security systems also, it has various real-time applications
such as surveillance, robotics, and biometrics.

```
Boundary boxes in object detection
```
## Introduction

### In computer vision, there are three types of tasks:

### 1. Image Classification:

```
Image Classification concerns with type or class of
the object in the given image. Output for this
algorithm is a class label for input photographs
```
### 2. Object localization:

```
Object localization is the primary concept behind all
image detection algorithms. Object localization refers
to identifying the location of one or more objects in an
image and drawing abounding box around their extent.
Instance segmentation is one of those ways of
localization. In instance segmentation, a bounding box
is created around the object, and a label to which that
object belongs is created at the top of the object.
Whereas in image segmentation, the boundary is
created around the object and it’s also pixel-wise. In-
stance segmentation creates square or rectangle
boundaries.
```
### 3. Object Detection:

```
Object detection is a computer vision technique to
identify various objects from an image or a video.
Object detection outputs class labels with proper
boundary boxes for objects in the image.
```

### Various algorithms can be used for Object

### Detection using Deep Learning some of them are:

### 1. YOLO:

```
YOLO uses convolutional neural networks for predict
ion of class labels and object’s location also.
YOLOv3 is more accurate and faster than previous
versions and SSD (Single Shot MultiBox Detector).
```
### YOLOv3:

YOLOv3 is a real-time, single-stage object detection model
that builds on YOLOv2 with several improvements.
Improvements include the use of a new backbone network,
Darknet-53 that utilizes residual connections, or in the words
of the author, "those newfangled residual network stuff", as
well as some improvements to the bounding box prediction
step, and use of three different scales from which to extract
features (similar to an FPN).

```
YOLOv3 Structure
```
### How does YOLO work?

YOLO looks at an image only once. And then applies only
one neural network on the image. The image is divided into
the SxS grid. Each cell can predict N bounding boxes. A
rectangle is formed from these bounding boxes to enclose
an object within it. So total, SxSxN boxes can be predicted
from the given image. Each bounding box has its own
predicted probabilities.
YOLO outputs a confidence score that tells us how
certain it is that the predicted bounding box encloses the
object. The task here is to categorize each object based on
the class shown in the image or the video. The output for a
given input image will be a confidence score that tells us
how certain it is that the predicted bounding box encloses
the object.

### 2. SSD:

```
SSD algorithms (Single Shot MultiBox Detector)
have a good balance between their accuracy and
speed. The SSD uses a small two-sided kernel on
the feature map created by CNN and image. By
doing so, it automatically predicts binding boxes
and segmentation opportunities.
An SSD works better on larger objects than on
smaller ones. We can say that its performance is
similar to the Faster-RCNN of the big objects.
```
### 3. Faster R-CNN:

```
Faster R-CNN consists of Region Proposal
Network (RPN) and Fast-RCNN. Anchor boxes
are introduced in faster-RCNN. There are three
types of anchor box- es, 128x128, 256x256, and
512x512. Additionally, it has three aspect ratios.
```
```
(1:1, 2:1, 1:2). This gives a total of 9 boxes to predict
the probability. This is the output of RPN, which is
given to Fast-RCNN. Here, the spatial pooling
technique is applied along with regression.
Faster R-CNN is 10 times faster than Fast R-CNN
for the same accuracy output.
```
```
Object detection with R-CNN
```
## Dataset Description

```
For this project, we used the dataset provided by our
professor, MS COCO (Microsoft common objects in
con- text) dataset.
```
- The dataset is created by gathering images of ordinary
    scenes, these scenes vary from normal to complex. These
    ordinary scenes in every single image contain basic items
    such as a person, car, animals, etc. in their common
    settings.
- It is large-scale object detection, segmentation, and
    captioning dataset. It has serval features like object
    segmentation, superpixel stuff segmentation. It also has
    over 330 thousand images and out of which more than
    200 thousand images are labeled.
- It also contains around 1.5 million instances, 80 object
    categories, 91 stuff categories, and at least 5 captions per
    single image which makes this dataset versatile.
- The training set(2017 Train Images 18GB in size)
    consists of around 118 thousand images along with
    annotations(2017 Train/Val annotations 241MB in size).
- Download data at [http://cocodataset.org/#download](http://cocodataset.org/#download)

## Project description

### Description:

```
In this project, we will implement a Deep Learning based Object
Detection System using the OpenCV library, MXNet framework,
and YoloV3 algorithm, to detect the instances of semantic objects
of a class human from an image or a video. To identify instances
of semantic objects from a video we ne ed to consider each
frame of a video as an individual image. We can also use the same
technique to detect the instances of semantic objects of other
classes such as cars, animals, food, etc. We will be using the MS
COCO dataset to train a model architecture that will take the
images and videos as the input and then detect the total number of
people present on a webcam or an image. The output will be a
window showing live footage of webcam on which at the bottom,
there will be a counter showing the total number of people present
```

on the footage, also there will be a boundary box around all
human class objects or any other class objects with a
confidence level.

### YOLOv3:

### Here are some concepts of the YOLOv

### algorithm:

```
Yolov3 Network Architecture
```
### Darknet architecture:

YOLOv3 uses darknet - 53 build to add a feature to the image.
As the name suggests it has 53 layers of communication. We
know that the number of layers indicates how deep the art of
construction is. YOLOv3 is a deeper builder than yolov2, as it
only goes to 19 people. There are 3x3 and 1x1 filters YOLOv
formats that help extract features from the image, in all
differentlayers.

```
Darknet-53 architecture
```
### Leaky ReLU:

```
We all know that ReLU (Rectified Linear Unit) is a
great tool for activating neural networks. But one step
ahead, Leaky ReLU is much better than ReLU itself.
One of the benefits of a rewarding ReLU is that it
almost eliminates the problem of “dying ReLU”.
ReLU problem death refers to the output provided by
the ReLU function for negative input (Always 0). In
the ReLU leak, there is a small slope with no incorrect
input values. That the output of those input values will
tend to be zero but not actually zero, which helps to
find the best neurons in each layer of the convolutional
```
```
network and therefore in the output results.
Here is a basic code to implement leaky ReLU in the
program and a chart to represent its functionality. On the
left-hand side is ReLU function, and the other chart
postulates leaky ReLU. The function, f(y) = ay is the slope
line for negative values, and in practice, the value of ‘a’ is
nearly around 0.01 – 0.001.
```
```
ReLU and leaky ReLU Function Graph
```
### Bounding boxes:

```
Bounding boxes are used to separate identified objects from
one another. Logistic regression is used to predict the
confidence score of the box.
```
### Anchor boxes:

```
In a single network layer, anchor boxes are responsible for
classification and prediction. It uses k means clustering method
for prediction.
```
### Batch normalization:

```
In general, we base our input data generally on activation tasks
or other strategies to improve the performance of our model. If
so, then can we disable our hidden layers to improve the
efficiency of the model? The familiarity of the cluster is used to
familiarize the input and concealment of the construction layers
in order to improve. Collectively, it improves accuracy by 2%,
and overuse can be reduced to a minimum.
```
### Accuracy threshold:

```
The main advantage of the model is that it produces bounding
boxes of partially visible objects (which do not need to be
identified) with less accuracy than other objects. After giving
the value of the accuracy limit to the model, it will not look for
items that offer less accuracy than the limit, resulting in better
availability.
```
### Configuration parameter:

```
Additionally, the model can predict more than one bounding
box for one object which sometimes overlaps on each other.
The configuration parameter is set as the number of spacing
boxes. We used different values in the parameter to see how
those boxes were made for different values. If we consider its
value as 1.0, then it will show all possible binding boxes
around the object. And at 0.1, it will delete some of the
```

```
required boxes from the account. So typically, 0.5 or 0.6 is
used for better finding and orderly extraction.
```
## Main references used for your project:

- “How to implement a YOLOv3 object detector from
    scratch in PyTorch” on medium.com, from this blog
    we get a knowledge about overall features and
    understanding of YOLOv3. Like, bounding boxes,
    improvements in YOLOv3 over YOLOv2, anchor
    boxes, loss function.
- We understand about how YOLOv3 works with
    darknet architecture, from this blog.
- Implementation of Keras model and object detection.

## Configuration parameter

```
In our project, there is one function that sets the
configuration parameter. What this parameter does is,
it generalizes the bounding boxes which generate
around one object only. For a single object, we get
multiple boxes. This parameter chooses the best box
from them which can get a label accordingly. In one of
those references, one person did not set the value for
the configuration parameter and gets the accuracy of
around 70%. But after implementing this parameter
our accuracy has risen by 8 to 10%.
```
### Accuracy Threshold value

```
While using different threshold values for accuracy,
we got different numbers of objects in each output.
This is be- cause, every time we change threshold
value, objects with accuracy below threshold value
will be dropped from final accuracy measurement and
output. Due to this, the model will not get the same
number of objects each time.
These are some results we got for the same image
with different threshold parameters, but with the same
configuration parameter value (0.5).
```
## Accuracy

## Threshold

## value

## Accuracy

## 60% (0.6) 98.96%

## 40% (0.4) 97.01%

## 9 0% (0. 9 ) 99.27%

## List of our contributions in the project:

- Data augmentation. (Augmented annotation files
    with dataset).
- With the threshold and configuration parameters
    ideal combination we took out the best accuracy.
- Trained the Keras model using YOLOv3 to gen-
    erate new weights to detect the objects in the im-
    age.
- Tried different activation functions to see the var-
    iation in the accuracy of the image detection.
       - Calculated an individual accuracy of each object in the
          image then took the average of accuracies of all the
          objects in that image to get a cumulative accuracy of
          all the objects in that image.
       - Increase in accuracy with respect to the original
          reference we used for our project.

## Analysis

### What did we do well?

```
At the starting phase of the project, we did not have any idea
about Keras, Matplotlib, YOLO models, TensorFlow, Neural
Networks, Activation Functions. But now we have a better
understanding of these terms and due to which we were able to
create a basic model to classify some objects from a given
image.
```
- We calculated the individual accuracy of and object in
    the image then took the average of accuracies of all the
    objects in the image to get a cumulative accuracy of an
    image.
- We used dataset utility to load datasets so that we can
    load data faster and get the results quicker.
- For plotting the labeled image, we used patches that is
    rectangular method.
- With the threshold and configuration parameters ideal
    combination we took out the best accuracy.
- Trained the Karas model using Yolov3 to generate new
    weights to detect the objects in the image.
- We also show bounding box on object in live video
    frame and counted the number of object with great
    accuracy.
- We can also detect other object by changing the label
    property in detection code.

### What could we have done better?

```
At the starting phase of the project, we were thinking about
implementing the project not only for webcam videos but also
detecting cars and people in the given video of city traffic. But
due to limited time, the growing difficulty of implementing the
project, and due to limited resources, we were unable to
implement this idea.
If we were able to use this item it would help in large areas of
computer vision such as surveillance systems, unmanned
vehicle systems etc.
```
### What is left for future work?

- From the present model we can only detect human

### from webcam frame.

- In future we can add other functionalities like

### calculating total number of cars in particular lines

### for traffic governance.

- For current COVID-19 situation, we can add more

### features like distance measure between two people

### which will be helpful in social distancing in crowded

### areas.

- One more idea is that, making a shopping
    application that detects the object and gives the best
    price, link to buy it from, and reviews of that item.


## Conclusion

Object detection is an important concept in the study
of robots and areas of computer vision. It requires a
great deal of attention to give superior results in the
better development of humans. However, there has
been a lot of research done and much more is still
going on in computer vision, not enough compared to
the technological advances of the 21st century. Object
detection can be used extensively in real-time
applications such as tracking applications, tracking
systems, pedestrian detection, and unmanned vehicle
programs.
But what we have learned so far in this project, is that
in object identification, YOLOv3 is a well-defined
method. It offers good accuracy even in small
databases, because it has deep architecture and a
complex model network.

## References

### 1. https://towardsdatascience.com/image-detection-

```
from-scratch-in-keras-f314872006c
```
### 2. https://medium.com/datadriveninvestor/yolov3-

```
from-scratch-using-pytorch-part1-474b49f7c8ef
```
### 3. https://towardsdatascience.com/creating-your-

```
own-object-detector-ad69dda69c
```
### 4. https://towardsdatascience.com/custom-object-

```
detection-using-tensorflow-from-scratch-
e61da2e
```
### 5. https://openaccess.thecvf.com/content_cvpr_2017/

```
papers/Chattopadhyay_Counting_Everyday_Objec
ts_CVPR_2017_paper.pdf.
```
### 6. https://arxiv.org/pdf/1804.02767.pdf

### 7. https://arxiv.org/pdf/1807.05511.pdf
