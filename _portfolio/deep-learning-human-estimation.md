---
title: "Deep Learning-Based Human Reconstruction"
excerpt: "A full pipeline that reconstructs a 3D human model based on deep learning"
header:
  image: /assets/images/human/cnn_front_img_to_mesh.jpg
---

This is a commercial project that I worked for a company in the field of virtual shopping. Due to the "agreement of disclosure", the code is not published and I can only explain briefly about the general idea of the project.

The slice-based methods in the field of human shape estimation are not effective because of problems like perspective distortion, pose variants and feature extraction, which is very error-prone and time-consuming. In contrast, the deep-learning based approaches are more simpler in term of deployment and it is also more robust under noisy conditions.

In this project, convolution neural networks are designed to take in human silhouettes and predict parameters for a statistical model of human shape. These parameters are then used to reconstruct the 3D corresponding mesh of the subject.

To create the training data, a combination of scripts and 3D software are used to generate thousands of pairs of silhouettes and the corresponding target statistical parameters.

In the training step, the model for front and side silhouettes are trained first and then another model is trained to combine pre-trained weights from the first two models. Training this way help the models learn distinguished features from the front and side silhouettes. In contrast, if the final model is trained from scratch with both front and side silhouettes, the trained weights will be more biased toward the front silhouettes, which ignore meaningful information in the side silhouettes.  

![cnn_front_img_to_mesh](/assets/images/human/cnn_front_img_to_mesh.jpg)

Regarding the learning aspect, this project helped me learn a lot more about the whole pipeline of machine learning-based project development. It's not just about training models, but about designing the whole pipeline from data generation to deployment, and about integration of different techniques from image processing to geometry processing that solve different tasks in the project.

For further questions, you can send me an email at khanhhh89@gmail.com.
Thanks for your time reading this
