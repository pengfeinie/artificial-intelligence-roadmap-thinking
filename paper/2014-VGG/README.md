**VGG Neural Networks: The Next Step After AlexNet.** AlexNet came out in 2012 and was a revolutionary advancement; it improved on traditional Convolutional Neural Networks (CNNs) and became one of the best models for image classification… until [VGG](https://arxiv.org/abs/1409.1556) came out.

**AlexNet.** When AlexNet was published, it easily won the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) and proved itself to be one of the most capable models for object-detection out there. Its key features include using ReLU instead of the tanh function, optimization for multiple GPUs, and overlapping pooling. It addressed overfitting by using data augmentation and dropout. So what was wrong with AlexNet? Well nothing was, say, particularly “wrong” with it. People just wanted even more accurate models.

**The Dataset.** The general baseline for image recognition is ImageNet, a dataset that consists of more than 15 million images labeled with more than 22 thousand classes. Made through web-scraping images and crowd-sourcing human labelers, ImageNet even hosts its own competition: the previously mentioned ImageNet Large-Scale Visual Recognition Challenge (ILSVRC). Researchers from around the world are challenged to innovate methodology that yields the lowest top-1 and top-5 error rates (top-5 error rate would be the percent of images where the correct label is not one of the model’s five most likely labels). The competition gives out a 1,000 class training set of 1.2 million images, a validation set of 50 thousand images, and a test set of 150 thousand images; data is plentiful. AlexNet won this competition in 2012, and models based off of its design won the competition in 2013.

## What is VGG ?

VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers. The “deep” refers to the number of layers with VGG-16 or VGG-19 consisting of 16 and 19 convolutional layers.

The VGG architecture is the basis of ground-breaking object recognition models. Developed as a deep neural network, the VGGNet also surpasses baselines on many tasks and datasets beyond ImageNet. Moreover, it is now still one of the most popular image recognition architectures.

Convolutional networks (ConvNets) currently set the state of the art in visual recognition. The aim of this project is to investigate how the ConvNet depth affects their accuracy in the large-scale image recognition setting.

Our main contribution is a rigorous evaluation of networks of increasing depth, which shows that a significant improvement on the prior-art configurations can be achieved by increasing the depth to 16-19 weight layers, which is substantially deeper than what has been used in the prior art. To reduce the number of parameters in such very deep networks, we use very small 3×3 filters in all convolutional layers (the convolution stride is set to 1).

### What is VGG16 ?

The VGG model, or VGGNet, that supports 16 layers is also referred to as VGG16, which is a convolutional neural network model proposed by A. Zisserman and K. Simonyan from the University of Oxford. These researchers published their model in the research paper titled, “[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).”

The VGG16 model achieves almost 92.7% top-5 test accuracy in ImageNet. ImageNet is a dataset consisting of more than 14 million images belonging to nearly 1000 classes. Moreover, it was one of the most popular models submitted to [ILSVRC-2014](http://www.image-net.org/challenges/LSVRC/2014/results). It replaces the large kernel-sized filters with several 3×3 kernel-sized filters one after the other, **thereby making significant improvements over AlexNet**. The VGG16 model was trained using Nvidia Titan Black GPUs for multiple weeks.

As mentioned above, the VGGNet-16 supports 16 layers and can classify images into 1000 object categories, including keyboard, animals, pencil, mouse, etc. Additionally, the model has an image input size of 224 x 224.

### What is VGG19 ?

The concept of the VGG19 model (also VGGNet-19) is the same as the VGG16 except that it supports 19 layers. The “16” and “19” stand for the number of weight layers in the model (convolutional layers). This means that VGG19 has three more convolutional layers than VGG16. We’ll discuss more on the characteristics of VGG16 and VGG19 networks in the latter part of this article.

## VGG – The Idea

Karen Simonyan and Andrew Zisserman proposed the idea of the VGG network in 2013 and submitted the actual model based on the idea in the 2014 ImageNet Challenge. They called it VGG after the department of Visual Geometry Group in the University of Oxford that they belonged to. 

So what was new in this model compared to the top-performing models AlexNet-2012 and ZFNet-2013 of the past years? First and foremost, compared to the large receptive fields in the first convolutional layer, this model proposed the use of a very small 3 x 3 receptive field (filters) throughout the entire network with the stride of 1 pixel. Please note that the receptive field in the first layer in AlexNet was 11 x 11 with stride 4, and the same was 7 x 7 in ZFNet with stride 2. 

The idea behind using 3 x 3 filters uniformly is something that makes the VGG stand out. Two consecutive 3 x 3 filters provide for an effective receptive field of 5 x 5. Similarly, three 3 x 3 filters make up for a receptive field of 7 x 7. This way, a combination of multiple 3 x 3 filters can stand in for a receptive area of a larger size. 

But then, what is the benefit of using three 3 x 3 layers instead of a single 7 x 7 layer? Isn’t it increasing the no. of layers, and in turn, the complexity unnecessarily? No. In addition to the three convolution layers, there are also three non-linear activation layers instead of a single one you would have in 7 x 7. This makes the decision functions more discriminative. It would impart the ability to the network to converge faster. 

Secondly, it also reduces the number of weight parameters in the model significantly. Assuming that the input and output of a three-layer 3 x 3 convolutional stack have C channels, the total number of weight parameters will be 3 * 32 C2 = 27 C2. If we compare this to a 7 x 7 convolutional layer, it would require 72 C2 = 49 C2, which is almost twice the 3 x 3 layers. Additionally, this can be seen as a regularization on the 7 x 7 convolutional filters forcing them to have a decomposition through the 3 x 3 filters, with, of course, the non-linearity added in-between by means of ReLU activations. This would reduce the tendency of the network to over-fit during the training exercise. 

Another question is – can we go lower than 3 x 3 receptive size filters if it provides so many benefits? The answer is “No.” 3 x 3 is considered to be the smallest size to capture the notion of left to right, top to down, etc. So lowering the filter size further could impact the ability of the model to understand the spatial features of the image.

The consistent use of 3 x 3 convolutions across the network made the network very simple, elegant, and easy to work with.

## **VGG Configurations**

The authors proposed various configurations of the network based on the depth of the network. They experimented with several such configurations, and the following ones were submitted during the ImageNet Challenge. 

A stack of multiple (usually 1, 2, or 3) convolution layers of filter size 3 x 3, stride one, and padding 1, followed by a max-pooling layer of size 2 x 2, is the basic building block for all of these configurations. Different configurations of this stack were repeated in the network configurations to achieve different depths. The number associated with each of the configurations is the number of layers with weight parameters in them. 

The convolution stacks are followed by three fully connected layers, two with size 4,096 and the last one with size 1,000. The last one is the output layer with Softmax activation. The size of 1,000 refers to the total number of possible classes in ImageNet.

VGG16 refers to the configuration “D” in the table listed below. The configuration “C” also has 16 weight layers. However, it uses a 1 x 1 filter as the last convolution layer in stacks 3, 4, and 5. This layer was used to increase the non-linearity of the decision functions without affecting the receptive field of the layer.

The nets are referred to their names (A-E). All configurations follow the generic design present in architecture and differ only in the depth: from 11 weight layers in the network A (8 conv. and 3 FC layers) to 19 weight layers in the network E (16 conv. and 3 FC layers). The width of conv. layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

In this discussion, we will refer to configuration “D” as VGG16 unless otherwise stated. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)

![img](https://lh3.googleusercontent.com/TtW43wmz1iQSVUnHZ7lMBwQ-PzogmkxpKskveXM40BoHC4JvJENw3ZXbgznlW1QSSgN1kx3BkwG8vrFo7NDGOLrMdAg7ie-GDOVr60G1ugqJ8KVOXRDGqZbno7aCyyyWqjf6qNjYaAtimfMu6w=s0)

The left-most “A” configuration is called VGG11, as it has 11 layers with weights – primarily the convolution layers and fully connected layers. As we go right from left, more and more convolutional layers are added, making them deeper and deeper. Please note that the ReLU activation layer is not indicated in the table. It follows every convolutional layer.

## VGG 16 Overall

**VGG16** is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to [ILSVRC-2014](http://www.image-net.org/challenges/LSVRC/2014/results). It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

![img](https://pengfeinie.github.io/images/vgg16-neural-network.jpg)

### **DataSet**

[ImageNet](http://www.image-net.org/) is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Amazon’s Mechanical Turk crowd-sourcing tool. Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held. ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. At all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images. ImageNet consists of variable-resolution images. Therefore, the images have been down-sampled to a fixed resolution of 256×256. Given a rectangular image, the image is rescaled and cropped out the central 256×256 patch from the resulting image.

### **VGG 16 Architecture**

Of all the configurations, VGG16 was identified to be the best performing model on the ImageNet dataset. Let’s review the actual architecture of this configuration. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)

 ![img](https://lh5.googleusercontent.com/yiz4POx7TGD21dQ7QvfI7fW5l4DNdvfR-EJsMmvxrKdeM9KNqz2TNWnOD7zmtIDXunVbx6zdXzQrL_6KD03QdIRauzwlaOidT9WIQA1O1NQ-M8_nqIes3hf_6SFVCQJQ2rKPWIkWearCBG5f1g=s0)

The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.

Three Fully-Connected (FC) layers follow a stack of convolutional layers (which has a different depth in different architectures): the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

All hidden layers are equipped with the rectification (ReLU) non-linearity. It is also noted that none of the networks (except for one) contain Local Response Normalisation (LRN), such normalization does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time.

The input to any of the network configurations is considered to be a fixed size 224 x 224 image with three channels – R, G, and B. The only pre-processing done is normalizing the RGB values for every pixel. This is achieved by subtracting the mean value from every pixel. 

Image is passed through the first stack of 2 convolution layers of the very small receptive size of 3 x 3, followed by ReLU activations. Each of these two layers contains 64 filters. The convolution stride is fixed at 1 pixel, and the padding is 1 pixel. This configuration preserves the spatial resolution, and the size of the output activation map is the same as the input image dimensions. The activation maps are then passed through spatial max pooling over a 2 x 2-pixel window, with a stride of 2 pixels. This halves the size of the activations. Thus the size of the activations at the end of the first stack is 112 x 112 x 64. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)![img](https://lh5.googleusercontent.com/f1H5NuiVDceNzWM73B4p1rBue91v84buLuNaXJzzKzOA8EORY3NT8m2Fff_PdfLcckThI64jBK2rqioC0cAVRoxPtW8JHY--XbG5H2wKtcm00mI6VwVfWiJlQL2WUv2b0SxJLqm6vLlRhtXN0w=s0)

![img](https://lh3.googleusercontent.com/g8BjMUQbAsvLuMSOq8BnWuKZQejwqrUjjrt-2zV-DfSHPwncvvvgJ6--odHBRgHO079U5p3gVhxlglMdzoKF32VjpCLngayjdG9aZOc7IJeepWn87E3VHzQTzWxOUPAd9Du56Q3ipSVzg4bViw=s0)

The activations then flow through a similar second stack, but with 128 filters as against 64 in the first one. Consequently, the size after the second stack becomes 56 x 56 x 128. This is followed by the third stack with three convolutional layers and a max pool layer. The no. of filters applied here are 256, making the output size of the stack 28 x 28 x 256. This is followed by two stacks of three convolutional layers, with each containing 512 filters. The output at the end of both these stacks will be 7 x 7 x 512.

The stacks of convolutional layers are followed by three fully connected layers with a flattening layer in-between. The first two have 4,096 neurons each, and the last fully connected layer serves as the output layer and has 1,000 neurons corresponding to the 1,000 possible classes for the ImageNet dataset. The output layer is followed by the Softmax activation layer used for categorical classification.



## References

- https://www.kaggle.com/blurredmachine/vggnet-16-architecture-a-complete-guide

- https://arxiv.org/abs/1409.1556
- https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/
- https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c
- https://www.robots.ox.ac.uk/~vgg/research/very_deep/
- https://neurohive.io/en/popular-networks/vgg16/
- https://github.com/ashushekar/VGG16

