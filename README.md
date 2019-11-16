# Assignment-1

Convolution: It is like moving along or across complete input object or screen in order to look at each and every part of input. Like a duster rubbing on board.


Filters/Kernels: Also called as feature extractor or 3x3 matrix that extracts out a given feature from a channel/image.


Epochs: When we go through our input training data again and again i.e. no. of times our model or algorithm we have generated will traverse the entire dataset.


1x1 Convolution: is a 1X! matrix that convolves over the image to get different edges of image to recognise it. It usually mixes 2 different channels to create a new channel. hence it links contextually linked features.


3x3 Convolution: is a 3x3 matrix that convolves over the image to get different edges of image to recognise it. This is usually preferred kernel as we can make any color out of it, its odd size hence helps to know axis of symmetry. Also it is a superset of all 2X2 matrices. It would always drop 2 pixels of image after convolving 3X3 kernel over image after diff. runs of convolultion.


Feature Maps: Collection of all features being together. Full channel is going to be a feature map.


Activation Function: They are used to introduce non-linearity to a neural network. If we don't use it then the network could become simple linear model. We need non-linearity in neural n/w because we want it to learn something complex and reoresent all non-linear mappings b/w i/p and o/p.


Receptive Field: RF of last layer of neural network must be atleast size of the object neural network is looking at. In order to recorgnise any image correctly the last layer of network must have looked at complete object/image.
