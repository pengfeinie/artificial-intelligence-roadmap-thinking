**VGG Neural Networks: The Next Step After AlexNet.** AlexNet came out in 2012 and was a revolutionary advancement; it improved on traditional Convolutional Neural Networks (CNNs) and became one of the best models for image classification… until [VGG](https://arxiv.org/abs/1409.1556) came out.

**AlexNet.** When AlexNet was published, it easily won the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) and proved itself to be one of the most capable models for object-detection out there. Its key features include using ReLU instead of the tanh function, optimization for multiple GPUs, and overlapping pooling. It addressed overfitting by using data augmentation and dropout. So what was wrong with AlexNet? Well nothing was, say, particularly “wrong” with it. People just wanted even more accurate models.

**The Dataset.** The general baseline for image recognition is ImageNet, a dataset that consists of more than 15 million images labeled with more than 22 thousand classes. Made through web-scraping images and crowd-sourcing human labelers, ImageNet even hosts its own competition: the previously mentioned ImageNet Large-Scale Visual Recognition Challenge (ILSVRC). Researchers from around the world are challenged to innovate methodology that yields the lowest top-1 and top-5 error rates (top-5 error rate would be the percent of images where the correct label is not one of the model’s five most likely labels). The competition gives out a 1,000 class training set of 1.2 million images, a validation set of 50 thousand images, and a test set of 150 thousand images; data is plentiful. AlexNet won this competition in 2012, and models based off of its design won the competition in 2013.

## What is VGG ?

VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers. The “deep” refers to the number of layers with VGG-16 or VGG-19 consisting of 16 and 19 convolutional layers.

The VGG architecture is the basis of ground-breaking object recognition models. Developed as a deep neural network, the VGGNet also surpasses baselines on many tasks and datasets beyond ImageNet. Moreover, it is now still one of the most popular image recognition architectures.

### What is VGG16 ?

The VGG model, or VGGNet, that supports 16 layers is also referred to as VGG16, which is a convolutional neural network model proposed by A. Zisserman and K. Simonyan from the University of Oxford. These researchers published their model in the research paper titled, “[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).”

The VGG16 model achieves almost 92.7% top-5 test accuracy in ImageNet. ImageNet is a dataset consisting of more than 14 million images belonging to nearly 1000 classes. Moreover, it was one of the most popular models submitted to [ILSVRC-2014](http://www.image-net.org/challenges/LSVRC/2014/results). It replaces the large kernel-sized filters with several 3×3 kernel-sized filters one after the other, **thereby making significant improvements over AlexNet**. The VGG16 model was trained using Nvidia Titan Black GPUs for multiple weeks.

As mentioned above, the VGGNet-16 supports 16 layers and can classify images into 1000 object categories, including keyboard, animals, pencil, mouse, etc. Additionally, the model has an image input size of 224 x 224.

### What is VGG19 ?

The concept of the VGG19 model (also VGGNet-19) is the same as the VGG16 except that it supports 19 layers. The “16” and “19” stand for the number of weight layers in the model (convolutional layers). This means that VGG19 has three more convolutional layers than VGG16. We’ll discuss more on the characteristics of VGG16 and VGG19 networks in the latter part of this article.





## References

- https://www.kaggle.com/blurredmachine/vggnet-16-architecture-a-complete-guide

- https://arxiv.org/abs/1409.1556
- https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/
- https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c

