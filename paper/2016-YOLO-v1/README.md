## YOLO(You Only Look Once)









https://scholar.google.com/

![image-20211109124400546](https://pengfeinie.github.io/images/image-20211109124400546.png)



[CVPR 2016 - YOLO v1 - You Only Look Once: Unified, Real-Time Object Detection](https://www.bilibili.com/video/BV1yP4y1G7n3/)

![](https://pengfeinie.github.io/images/2021-11-17_134927.jpg)

Before diving into YOLO, we need to go through some terms:

**1-Intersect Over Union (IOU):**



**2-** **Precision:**

Simply we can define precision as the ratio of **true** positive(true predictions) (TP) and the total number of **predicted** positives(total predictions). The formula is given as such:

![img](https://miro.medium.com/max/60/1*-Y2bLNfGDkcZ5BSPPzHBjQ.png?q=20)

![img](https://miro.medium.com/max/180/1*-Y2bLNfGDkcZ5BSPPzHBjQ.png)

For example, imagine we have 20 images, and we know that there are 120 cars in these 20 images.

Now, let’s suppose we input these images into a model, and it detected 100 cars (here the model said: I’ve found 100 cars in these 20 images, and I’ve drawn bounding boxes around every single car of them).

To calculate the precision of this model, we need to check the 100 boxes the model had drawn, and if we found that 20 of them are incorrect, then the precision will be =80/100=0.8

**3-Recall:**

If we look at the precision example again, we find that it doesn’t consider the total number of cars in the data (120), so if there are 1000 cars instead of 120 and the model output 100 boxes with 80 of them are correct, then the precision will be 0.8 again.

To solve this, we need to define another metric, called the **Recall,** which is the ratio of **true** positive(true predictions) and the total of ground truth positives(total number of cars). The formula is given as such:

![img](https://miro.medium.com/max/60/1*nx6V3Q_EqnGWzcLfW_lL-A.png?q=20)

![img](https://miro.medium.com/max/198/1*nx6V3Q_EqnGWzcLfW_lL-A.png)

For our example, the recall=80/120=0.667.

Now we can notice that the recall measures how well we detect **all** the objects in the data.

![img](https://miro.medium.com/max/350/1*kaqtNALKZujx1FGlbK11OQ.png)

**4- Average Precision and Mean Average Precision(mAP):**

A brief definition for the Average Precision is the **area** under the **precision-recall curve.**

**AP** combines both precision and recall together. It takes a value between 0 and 1 (higher is better). To get **AP** =1 we need both the precision and recall to be equal to 1. The **mAP** is the mean of the AP calculated for all the classes.







Before we go into YOLOs details we have to know what we are going to predict. Our task is to predict a class of an object and the bounding box specifying object location. Each bounding box can be described using four descriptors:

1. center of a bounding box (**bx**,**by**)
2. width (**bw**)
3. height (**bh**)
4. value **c** is corresponding to a class of an object (f.e. car, traffic lights,…).

We’ve got also one more predicted value pc which is a probability that there is an object in the bounding box, I will explain in a moment why do we need this.

![img](https://pengfeinie.github.io/images/bbox-1.png)

YOLO uses a single bounding box regression to predict the height, width, center, and class of objects. In the image above, represents the probability of an object appearing in the bounding box.

## YOLO v1 The Architecture

![image-20211109160501058](https://pengfeinie.github.io/images/image-20211109160501058.png)

[YOLO V1 Paper](https://arxiv.org/pdf/1506.02640.pdf)

**The Model**. Our system models detection as a regression problem. It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes, and C class probabilities. These predictions are encoded as an S × S × (B ∗ 5 + C) tensor. For evaluating YOLO on [PASCAL VOC](https://paperswithcode.com/dataset/pascal-voc), we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor. We only predict one set of class probabilities per grid cell, regardless of the number of boxes B. 

*The PASCAL Visual Object Classes (VOC) 2012 dataset contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks. The PASCAL VOC dataset is split into three subsets: 1,464 images for training, 1,449 images for validation and a private testing set.*

For each grid cell,

- it predicts **B** boundary boxes and each box has one **bounding box confidence score**,
- it detects **one** object only regardless of the number of boxes B,
- it predicts **C** **conditional class probabilities** (one per class for the likeliness of the object class).

To evaluate [PASCAL VOC](https://paperswithcode.com/dataset/pascal-voc), YOLO uses 7×7 grids (S×S), 2 boundary boxes (B) and 20 classes (C).

Let’s get into more details. Each boundary box contains 5 elements: (*x, y, w, h*) and a **bounding box confidence score**. The confidence score reflects how likely the box contains an object (**objectness**) and how accurate is the boundary box. We normalize the bounding box width *w* and height *h* by the image width and height. *x* and *y* are offsets to the corresponding cell. Hence, *x, y, w* and *h* are all between 0 and 1. Each cell has 20 conditional class probabilities. The **conditional class probability** is the probability that the detected object belongs to a particular class (one probability per category for each cell). So, YOLO’s prediction has a shape of (S, S, B×5 + C) = (7, 7, 2×5 + 20) = (7, 7, 30).

![img](https://pengfeinie.github.io/images/output_tensor.png)

The architecture of YOLO v1 is not complicated, in fact it's just a convolutional backbone with two fully connected layers, much like an image classification network architecture. The clever part of YOLO (the part that makes it an object detector) is in the interpretation of the outputs of those fully connected layers. However, the concepts underlying that interpretation are complex, and can be difficult to grasp on a first reading. 

The authors designed their own convolutional backbone which was inspired by GoogLeNet. But I just want to point that it's just a feature extractor, and you could swap in any backbone you like, and as long as you made the size of the fully connected layers line up, it would all work fine. I won't dwell on the backbone any longer, the object detection is all done in the head.

As I said earlier, the network architecture is very simple, it's just a backbone with two fully connected layers. Let's blow up that last layer in a bit more detail. I'm going to refer to it as the *output tensor* to make it easier to refer to.

The first thing you might notice is that I've been calling it a fully connected layer, but it sure doesn't look like one. Don't let the 3D shape fool you, it *is* fully connected, it is *not* produced by a convolution, they just reshape it because it's easier to interpret in 3D. If implemented in PyTorch, you can imagine it being coded as a fully connected layer that is then reshaped into a 3D tensor. Alternatively, you can imagine unrolling the 3D tensor into one long vector of length `1470 (7x7x30)`. However you imagine it, it is fully connected, every output neuron is connected to every neuron in the 4096-vector before it.

So why reshape it into 3D? What do all those outputs mean? Why do those outputs make it an object detector? I'll start with the reason that it's `7x7`. To clarify my notation and make it easier to talk about, I will refer to a *cell*, and what I mean by that is a single position in the `7x7` grid of the output tensor. Therefore each cell is a single vector of length 30, I have highlighted one such cell in the diagram.

In order to predict a single box, the network must output a number of things. Firstly it must encode the coordinates of the box which YOLO encodes as `(x, y, w, h)`, where `x` and `y` are the center of the box. Early I suggested you familiarise yourself with box parameterisation, because YOLO does output the actual coordinates of the box. Firstly, the width and height are normalised with respect to the image width, so if the network outputs a value of `1.0` for the width, it's saying the box should span the entire image, likewise `0.5` means it's half the width of the image. Note that the width and height have nothing to do with the actual grid cell itself. The `x and y` values *are* parameterised with respect to the grid cell, they represent offsets from the grid cell position. The grid cell has a width and height which is equal to `1/S` (we've normalised the image to have width and height 1.0). If the network outputs a value of `1.0` for `x`, then it's saying that the `x` value of the box is the `x` position of the grid cell plus the width of the grid cell.

Secondly, YOLO also predicts a confidence score for each box which represents the probability that the box contains an object. Lastly, YOLO predicts a class, which is represented by a vector of `C` values, and the predicted class is the one with the highest value. Now, here's the catch. YOLO does *not* predict a class for every box, it predicts a class *for each cell*. But each cell is associated with two boxes, so those boxes will have the same predicted class, even though they may have different shapes and positions. 

So each cell is responsible for predicting boxes from a single part of the image. More specifically, each cell is responsible for predicting precisely two boxes for each part of the image. Note that there are 49 cells, and each cell is predicting two boxes, so the whole network is only going to predict 98 boxes. That number is fixed.

## Training

This section explains how YOLO performs unified detection. Without a selective search algorithm to propose bounding boxes, YOLO divides the entire image into a grid and predicts bounding boxes at each grid cell. The predictions at each grid cell contains all the revelant information needed to localise and determine object classes in the image. This is explained in greater detail as a series of the following steps

- An object center is assigned to each object in a given image. This center is chosen by dividing the input image into an **S*×*S** grid. If the center of an object in the image falls into a particular grid cell then that grid cell is responsible for detecting that object. In the **448×448** image below, the image is divided into a **7×7** grid (S = 7). As there is only 1 object, a bird, in this image, the grid cell responsible for predicting the bird is the **(4,4)** grid cell. This was chosen based on the cross center of the bounding box drawn over the bird falling in that position. [source](https://araintelligence.com/blogs/deep-learning/object-detection/yolo_v1)

  ![How Object Centers are assigned](https://cdn.araintelligence.com/images/object-detection/Grid_Description.png)

- The CNN outputs a prediction tensor of the size  **S*×*S*×(*Bx5+C)** where S are the spatial dimensions, B is the number of bounding box predictions you want at each cell (the bounding box with the highest confidence is chosen for the final prediction), C is the number of classes in your dataset and 5 represents the bounding box coordinates along with the confidence of the classifier. The number of bounding boxes, B, you want at each cell can be arbitrary and more bounding boxes makes the model slightly more accurate but remember that increasing B also increases the computational and memory costs.

- Each grid cell prediction contains B number of bounding boxes and C class predictions. A bounding box prediction consists of 5 elements: The x,y,w,h values of the bounding box and p, the probability that there is an object whose center falls within this grid cell (this is different from what is in C, which is the probability that the object belongs to a certain class). [source](https://araintelligence.com/blogs/deep-learning/object-detection/yolo_v1) The x,y coordinates of the bounding box are **predicted relative to the grid cell** and not the entire image but the width and height (w,h) are **predicted relative to the entire image**. Each grid cell also predicts the probability of the detected object belonging to a specific class class_i given that it is confident there is an object whose's center is in the grid cell and this information is all contained in C.

  ![Grid Cell Description](https://cdn.araintelligence.com/images/object-detection/grid_cell.png)

- For use later in loss calculations, the ground truth bounding box cordinates are all parameterised to be between 0 and 1. The width and height of the bounding box are calculated as a ratio of the entire image's width and height. The (x,y) grid cell offsets are parameterised as a ratio of a grid's width and height. For example, given the a 448×448 image and S = 7. By dividing the width/height by the grid size, we have a grid dimension where each grid cell is 64×64 pixels wide. Therefore a bounding box from our dataset with x = 32, y = 16, width = 300 and height = 150, would be parameterised as:

  ![](https://pengfeinie.github.io/images/xywh.bmp)

### Using the PascalVOC dataset for object detection

The [PascalVOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), is a dataset for object detection, classification and segmentation. The total size on disk is about 5.17GB for the 2007 and 2012 dataset, which makes it perfect for grokking the performance of the YOLOv1 algorithm.

A sample image with its corresponding annotation file is shown below, the parts of the annotation file to observe have been highlighted in red.

![Pascal VOC 2012 - 2007_000676 image](https://cdn.araintelligence.com/images/object-detection/2007_000676.jpg)

![Pascal VOC 2012 - 2007_000676 XML Annotation](https://cdn.araintelligence.com/images/object-detection/2007_000676_annotation.png)

The object class is denoted in 'annotation -> object -> name' tag, and this would be one of the 20 classes in the Pascal VOC dataset ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"). \ The bounding boxes are encoded in the top-left and bottom-right coordinates (min-max encoding), however, based on the YOLO algorithm, predictions are made with reference to the object centers (center encoding) and not their min-max coordinates. Therefore, we need to convert from the min-max encoding in the annotation file to the bounding box center encoding for use in the YOLO algorithm.

**Converting from Bounding box min-max encoding to Bounding box center encoding**

To make consequent calculations easier, we normalise the bounding box coordinates to values between 0 and 1 i.e b*∈[0,1]4×1. Let the bounding box coordinates in min-max encoding be defined as b{min-max} and the bounding box coordinates in center encoding be defined as b_{center}.

Where W,H are the width and height of the image respectively, Converting from b_{min-max} to b_{center}can simply be defined as

![](https://pengfeinie.github.io/images/xywbcontert.bmp)

This is easily implemented as:

```python
def convert(size, box):
    W = size[0]
    H = size[1]
    x_min, x_max, y_min, y_max = box

    x = (x_min + x_max)/2.0 * (1/W)
    y = (y_min + y_max)/2.0 * (1/H)
    w = (x_max - x_min) * (1/W)
    h = (y_max - y_min) * (1/H)

    return (x,y,w,h)
```

**Converting from center bounding box encoding to YOLO bounding box encoding**

![image-20211117190423273](https://pengfeinie.github.io/images/image-20211117190423273.png)

We currently have b_{center} which contains the x,y location of the object center and the width and height of the object, scaled between 0 and 1.

To convert this to YOLO bounding box encoding (b_{yolo}) which takes the x,y location of the object center (relative to the image's width and height) and **converts it to an (x,y) coordinate relative to the grid cell** (fig 3). The width and height are left as they are in the YOLO encoding.

This is formalised as

![image-20211117190546310](https://pengfeinie.github.io/images/image-20211117190546310.png)

During the loss calculations, the square root of the width, sqrt{w} and height, sqrt{h} are used as they ensure stability and prevent the loss function from penalising small width and height predictions.



This section covers how I trained the YOLOv1 network, the missing pieces of the puzzle yet to be covered are how to choose the "best" bounding boxes amongst the ones predicted at each grid cell and the loss function used for training the network.

I trained the model on the Pascal VOC 2007+2012 dataset. Here I've set the S*×*S*, spatial values to 7, which gives us a 7×7 grid. There are 20 classes in the VOC dataset and C = 20. Per the YOLO paper, we also set the number of bounding boxes per grid cell, B, to 2. Which means the predicted tensor from the CNN would be 7×7×30.

#### Bounding Box Selection Strategy

The output prediction tensor from the YOLO model is of size S*∗*S*∗(*B*∗5+*C*). In this section, we still assume S=7,B=2 and C=20.![image-20211118132118780](https://pengfeinie.github.io/images/image-20211118132118780.png)

How do we decide which bounding box to use for the loss calculations at a particular cell? Let the two bounding boxes be represented as \hat{b1} and \hat{b2} and let the ground truth bounding box at that grid cell be b*b*. The bounding box chosen is simply the one that has the maximum intersection over union with the ground truth. i.e

![image-20211118134213789](E:\npfsourcecode\java\sourcecode\pengfeinie.github.io\images\image-20211118134213789.png)

#### Intersection over Union (IoU)

IOU can be computed as Area of Intersection divided over Area of Union of two boxes, so IOU must be ≥0 and ≤1.![img](https://pengfeinie.github.io/images/iou.png)

*the above image from [source](https://amrokamal-47691.medium.com/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899) .* When predicting bounding boxes, we need the find the IOU between the predicted bounding box and the ground truth box to be ~1.

![img](https://pengfeinie.github.io/images/cat.jpeg)

In the left image, IOU is very low, but in the right image, IOU is ~1.

This is also known as the Jacquard Index and it is simply a measure of how similar the predicted bounding box is to the ground truth bounding box.

This function is very important and it plays a HUGE role in our accurate our model is. In the context of the YOLO algorithm, it is what we use to select the best predicted bounding for use in loss calculations.

![IoU illustration](https://pengfeinie.github.io/images/iou1.jpg)

Using the diagram above, the IoU is simply the blue region as this is the area common to the green bounding box, A (ground truth) and the red bounding box, B (prediction). To give this a formal definition, it is the ratio of the area common to both boxes (blue region) to the total area of both bounding boxes.

Let the green and red bounding boxes be A, B respectively. Then their top-left and bottom right coordinates are defined as





For our discussion, we crop our original photo. YOLO divides the input image into an **S**×**S** grid. Each grid cell predicts only **one** object. For example, the yellow grid cell below tries to predict the “person” object whose center (the blue dot) falls inside the grid cell. Each grid cell detects only one object. [source](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

![img](https://miro.medium.com/max/700/1*6qZXYCDUkC5Bc8nRolT0Mw.jpeg)

Each grid cell predicts a fixed number of boundary boxes. In this example, the yellow grid cell makes two boundary box predictions (blue boxes) to locate where the person is. Each grid cell make a fixed number of boundary box guesses for the object. [source](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

![img](https://pengfeinie.github.io/images/box.jpeg)

However, the one-object rule limits how close detected objects can be. For that, YOLO does have some limitations on how close objects can be. For the picture below, there are 9 Santas in the lower left corner but YOLO can detect 5 only. YOLO may miss objects that are too close.

![img](https://miro.medium.com/max/700/1*j4PnWfxP3yoVPOFyI27tww.jpeg)







The **class confidence score** for each prediction box is computed as:

![img](https://miro.medium.com/max/700/1*qVL77IZyEnra4DvENayXUA.png)

It measures the confidence on both the classification and the **localization** (where an object is located).

We may mix up those scoring and probability terms easily. Here are the mathematical definitions for your future reference.

![img](https://miro.medium.com/max/700/1*0IPktA65WxOBfP_ULQWcmw.png)



The first five values encode the location and confidence of the first box, the next five encode the location and confidence of the next box, and the final 20 encode the 20 classes (because Pascal VOC has 20 classes). In total, the size of the vector is `5xB + C` where `B` is the number of boxes, and `C` is the number of classes.







For every grid cell, you will get two bounding boxes, which will make up for the starting 10 values of the 1*30 tensor. The remaining 20 denote the number of classes. The values denote the class score, which is the conditional probability of object belongs to class i, if an object is present in the box.

![](https://pengfeinie.github.io/images/yolov1_grid1.jpg)

![](https://pengfeinie.github.io/images/yolo1.gif)

Next, we multiply all these class score with bounding box confidence and get class scores for different bounding boxes. 

![](https://pengfeinie.github.io/images/yolo1_grid.gif)

We do this for all the grid cells. That is equal to 7 x 7 x 2 = 98.

![](https://pengfeinie.github.io/images/yolo1_all_grid.gif)









![这里写图片描述](https://pengfeinie.github.io/images/yolo1_predict1.gif)













The Yolo was one of the first deep, one-stage detectors and since the first paper was published in **CVPR 2016**, each year has brought with it a new Yolo paper or tech report. We begin with Yolo v1 [1], but since we are primarily interested in analyzing loss functions, all we really need to know about the Yolo v1 CNN **(Figure 2a)**, is that is takes an RGB image (**448×448×3**) and returns a cube (**7×7×30**), interpreted in **(Figure 2b)**.

![](https://pengfeinie.github.io/images/00adc0adec6423a45a0706a4ce2dc01d.png)

#### YOLO v3

Download: https://pjreddie.com/media/files/yolov3.weights and move to under cfg folder.









## References

- [https://www.mathworks.com/discovery/object-detection.html](https://www.mathworks.com/discovery/object-detection.html)
- [https://paperswithcode.com/task/object-detection](https://paperswithcode.com/task/object-detection)
- [https://www.datacamp.com/community/tutorials/object-detection-guide](https://www.datacamp.com/community/tutorials/object-detection-guide)
- [https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html](https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html)
- [https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/](https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/)
- [Object Detection in 20 Years: A Survey](https://www.semanticscholar.org/paper/Object-Detection-in-20-Years%3A-A-Survey-Zou-Shi/bd040c9f76d3b0b77e2065089b8d344c9b5d83d6#extracted)  
- [https://arxiv.org/pdf/1905.05055.pdf](https://arxiv.org/pdf/1905.05055.pdf)
- [https://pengfeinie.github.io/files/1905.05055.pdf](https://pengfeinie.github.io/files/1905.05055.pdf) 
- [https://link.springer.com/article/10.1007/s11263-019-01247-4](https://link.springer.com/article/10.1007/s11263-019-01247-4)
- [https://machinelearningmastery.com/object-recognition-with-deep-learning/](https://machinelearningmastery.com/object-recognition-with-deep-learning/)
- [https://viso.ai/deep-learning/object-detection/](https://viso.ai/deep-learning/object-detection/)
- https://paperswithcode.com/dataset/pascal-voc
- https://www.harrysprojects.com/articles/yolov1.html
- https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c
- https://amrokamal-47691.medium.com/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899
- https://blog.csdn.net/hrsstudy/article/details/70305791?spm=1001.2014.3001.5501
- https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
- https://araintelligence.com/blogs/deep-learning/object-detection/yolo_v1
- http://yann.lecun.com/exdb/lenet/
