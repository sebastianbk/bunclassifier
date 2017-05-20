# Bun Classifier based on VGG16

This project trains a Convolutional Neural Network based on VGG16 to classify whether a picture of a bun is that of a plain bun or a chocolate bun.

Find the sample dataset here: [https://drive.google.com/file/d/0B0R1uCX3eS6teWk5aFpvOUlnVnM/view?usp=sharing](https://drive.google.com/file/d/0B0R1uCX3eS6teWk5aFpvOUlnVnM/view?usp=sharing)

The dataset includes just 128 images. However, by using the VGG16 network with its trained weights (from [ImageNet](http://www.image-net.org/)), it is possible to train a network that offers a high level of accuracy (97.85% efter just one epoch with 64 steps).

## Notes

I trained this network as part of taking the *fast.ai* course in deep learning. Check it out here: (http://course.fast.ai/)[http://course.fast.ai/]