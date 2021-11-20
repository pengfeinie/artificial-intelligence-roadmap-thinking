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

## **VGG Configurations**

The authors proposed various configurations of the network based on the depth of the network. They experimented with several such configurations, and the following ones were submitted during the ImageNet Challenge. 

A stack of multiple (usually 1, 2, or 3) convolution layers of filter size 3 x 3, stride one, and padding 1, followed by a max-pooling layer of size 2 x 2, is the basic building block for all of these configurations. Different configurations of this stack were repeated in the network configurations to achieve different depths. The number associated with each of the configurations is the number of layers with weight parameters in them. 

The convolution stacks are followed by three fully connected layers, two with size 4,096 and the last one with size 1,000. The last one is the output layer with Softmax activation. The size of 1,000 refers to the total number of possible classes in ImageNet.

VGG16 refers to the configuration “D” in the table listed below. The configuration “C” also has 16 weight layers. However, it uses a 1 x 1 filter as the last convolution layer in stacks 3, 4, and 5. This layer was used to increase the non-linearity of the decision functions without affecting the receptive field of the layer.

In this discussion, we will refer to configuration “D” as VGG16 unless otherwise stated. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)

![img](https://lh3.googleusercontent.com/TtW43wmz1iQSVUnHZ7lMBwQ-PzogmkxpKskveXM40BoHC4JvJENw3ZXbgznlW1QSSgN1kx3BkwG8vrFo7NDGOLrMdAg7ie-GDOVr60G1ugqJ8KVOXRDGqZbno7aCyyyWqjf6qNjYaAtimfMu6w=s0)

The left-most “A” configuration is called VGG11, as it has 11 layers with weights – primarily the convolution layers and fully connected layers. As we go right from left, more and more convolutional layers are added, making them deeper and deeper. Please note that the ReLU activation layer is not indicated in the table. It follows every convolutional layer.

## **VGG 16 Architecture**

Of all the configurations, VGG16 was identified to be the best performing model on the ImageNet dataset. Let’s review the actual architecture of this configuration. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)

 ![img](https://lh5.googleusercontent.com/yiz4POx7TGD21dQ7QvfI7fW5l4DNdvfR-EJsMmvxrKdeM9KNqz2TNWnOD7zmtIDXunVbx6zdXzQrL_6KD03QdIRauzwlaOidT9WIQA1O1NQ-M8_nqIes3hf_6SFVCQJQ2rKPWIkWearCBG5f1g=s0)

The input to any of the network configurations is considered to be a fixed size 224 x 224 image with three channels – R, G, and B. The only pre-processing done is normalizing the RGB values for every pixel. This is achieved by subtracting the mean value from every pixel. 

Image is passed through the first stack of 2 convolution layers of the very small receptive size of 3 x 3, followed by ReLU activations. Each of these two layers contains 64 filters. The convolution stride is fixed at 1 pixel, and the padding is 1 pixel. This configuration preserves the spatial resolution, and the size of the output activation map is the same as the input image dimensions. The activation maps are then passed through spatial max pooling over a 2 x 2-pixel window, with a stride of 2 pixels. This halves the size of the activations. Thus the size of the activations at the end of the first stack is 112 x 112 x 64. [source](https://www.mygreatlearning.com/blog/introduction-to-vgg16/)![img](https://lh5.googleusercontent.com/f1H5NuiVDceNzWM73B4p1rBue91v84buLuNaXJzzKzOA8EORY3NT8m2Fff_PdfLcckThI64jBK2rqioC0cAVRoxPtW8JHY--XbG5H2wKtcm00mI6VwVfWiJlQL2WUv2b0SxJLqm6vLlRhtXN0w=s0)

![img](https://lh3.googleusercontent.com/g8BjMUQbAsvLuMSOq8BnWuKZQejwqrUjjrt-2zV-DfSHPwncvvvgJ6--odHBRgHO079U5p3gVhxlglMdzoKF32VjpCLngayjdG9aZOc7IJeepWn87E3VHzQTzWxOUPAd9Du56Q3ipSVzg4bViw=s0)

The activations then flow through a similar second stack, but with 128 filters as against 64 in the first one. Consequently, the size after the second stack becomes 56 x 56 x 128. This is followed by the third stack with three convolutional layers and a max pool layer. The no. of filters applied here are 256, making the output size of the stack 28 x 28 x 256. This is followed by two stacks of three convolutional layers, with each containing 512 filters. The output at the end of both these stacks will be 7 x 7 x 512.

The stacks of convolutional layers are followed by three fully connected layers with a flattening layer in-between. The first two have 4,096 neurons each, and the last fully connected layer serves as the output layer and has 1,000 neurons corresponding to the 1,000 possible classes for the ImageNet dataset. The output layer is followed by the Softmax activation layer used for categorical classification.



## Results

### Models

We release our two best-performing models, with 16 and 19 weight layers (denoted as configurations *D* and *E* in the [publication](https://www.robots.ox.ac.uk/~vgg/research/very_deep/#pub)). The models are released under [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/). Please cite our technical report if you use the models.





## References

- https://www.kaggle.com/blurredmachine/vggnet-16-architecture-a-complete-guide

- https://arxiv.org/abs/1409.1556
- https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/
- https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c
- https://www.robots.ox.ac.uk/~vgg/research/very_deep/

