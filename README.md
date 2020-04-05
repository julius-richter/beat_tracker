# Style-specific Beat Tracking with Deep Neural Networks 

This repository contains the code to my [Master thesis](https://www2.ak.tu-berlin.de/~akgroup/ak_pub/abschlussarbeiten/2019/Richter_MasA.pdf) on beat tracking.

## Abstract 

In this thesis, a computational method for extracting the beat positions from audio signals is presented. The proposed beat tracking system is based on temporal convolutional networks which capture the sequential structure of audio input. A dynamic Bayesian network is used to model beat periods of various lengths and align the predicted beat positions to the best global solution. The system is evaluated on four datasets of various musical genres and styles and achieves state-of-the-art performance. Compared against a current state-of-the-art beat tracker, the proposed approach maintains competitive performance but with two distinct computational advantages. It works causally and requires considerably less learnable parameters. In addition, due to the highly parallelizable structure of convolutional neural networks, computational efficiency dramaticallyincreases when training on GPUs.

