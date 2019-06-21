---
title: Crack Segmentation
excerpt: deep learning models to detect/segment crack for a bridge investigation project
header:
  image: /assets/images/crack/crack_segmentation.png
---

This is a project I worked in cooperation with the computer vision lab at The University of Bauhaus. Due to the lack of training data, I firstly approached the crack detection problem as a classification problem based on patch. Specifically, crack images are subdivided into patches, which are label as having crack pixels or not. These positive/negative image patches are used to train models like VGG16, Resnet, etc. However, even that the trained could reach very high accuracy on the test set, it performs poorly in real-world condition. The reason is that by dividing the images into patches in an effort to increase training dataset, we lose the contextual information of each crack. This make the model sensitive to noise.

I then decided to change the problem as crack pixel segmentation. Using LabelMe as a marking tool, I labeled over 500 crack images, merged from different crack dataset and our dataset at the computer vision lab. Finally, with some resampling and augmentation, we have over 10.000 crack images labeled at pixel level. it turns our that the model works very well in corner cases in real situations, as shown in my github page.

Fore more information about the project, please check [my github project](https://github.com/khanhha/crack_segmentation). For convenience, below is a copy paste of the introduction from the project page.

# Crack Segmentation

Here I present my solution to the problem crack segmentation for both pavement and concrete meterials. In this article, I describe the approaches, dataset that I exprimented with and desmonstrate the result. My approach is based on the UNet network with transfer learning on the two popular architectures: VGG16 and Resnet101. The result shows that a large crack segmentation dataset helps improve the performance of the model in diverse cases that
could happen in practice.

# Contents
 - [Inference Result Preview](#Inference-Result-Preview)
 - [Overview](#Overview)
 - [Dataset](#Dataset)
 - [Dependencies](#Dependencies)
 - [Test Images Collection](#Test-Images-Collection)
 - [Inference](#Inference)
 - [Training](#Training)
 - [Result](#Result)
 - [Citation](#Citation)

# Inference result preview
Below are the results from several test cases. For more test case results, please see images under the folder ./test_results

![](/assets/images/crack/show_result_2.jpg)

![](/assets/images/crack/show_result_3.jpg)

![](/assets/images/crack/show_result_4.jpg)

# Overview
Crack segmentation is an important task in structure investigation problems. For example, in the bridge investigation project, a done is controlled to fly around bridges to take pictures of different bridge surfaces. The pictures will be then processed by computer to detect potential regions on the bridge surface that might be damaged. The more accurate the model is, the less human effort we need to process these images. Otherwise, the operators will have to check every single image, which is boring and error-prone. One challenge in this task is that the model is sentisive to noise and other objects such as moss on crack, title lines, etc. In this project, I tried to label over 300 high-resolution images from the crack dataset at my university and merged over 9 different segmentation crack datasets available on the Internet. The result show that the model could be able to distinguish crack from tree, title linesand other different noise in reality.

# Dataset
From my knowledge, the dataset used in the project is the largest crack segmentation dataset so far. It contains around 11.200 images that are merged from 12 available crack segmentation datasets. The name prefix of each image is assigned to the corresponding dataset name that the image belong to. There're also images with no crack pixel, which could be filtered out by the file name pattern "noncrack*" All the images are resized to the size of (448, 448). The two folders images and masks contain all the images. The two folders train and test contain training and testing images splitted from the two above folder. The splitting is stratified so that the proportion of each dataset in the train and test folder are similar

If you want access to the original datasets before they are merged, please contact me through email: khanhhh89@gmail.com

***
# Dependencies
```python
conda create --name crack
conda install -c anaconda pytorch-gpu
conda install -c conda-forge opencv
conda install matplotlib scipy numpy tqdm pillow
```

***
# Inference
- download the pre-trained model [unet_vgg16](https://drive.google.com/open?id=1wA2eAsyFZArG3Zc9OaKvnBuxSAPyDl08) or
[unet_resnet_101]().
- put the downloaded model under the folder ./models
- run the code
```pythonstub
python inference_unet.py  -in_dir ./test_images -model_path ./models/model_unet_resnet_101_best.pt -out_dir ./test_result
```

***
# Test Images Collection
The model works quite well in situations where there are just almost crack pixels and the concrete background in the images. However, it's often not the case in reality, where lots of different objects could simultenously show up in an image. Therefore, to evaluate the robustness of the crack model, I tried to come up with several cases that could happen in practice. These images could be found in the folder ./test_imgs in the same repository

- pure crack: these are ideal cases where only crack objects occur in the images.
- like crack: pictures of this type contains details that look like crack
- crack with moss: there is moss on crack. These cases occur a lot in reality.
- crack with noise: the background (wall, concrete) are lumpy  
- crack in large context: the context is large and diverse. For example, the whole house or street with people

| | | |
|------------------|--------|--------------|
| pure crack | like crack | crack with moss |
| ![pure crack](/assets/images/crack/pure_crack.jpg) | ![](/assets/images/crack/like_crack.jpg)| ![](/assets/images/crack/crack_with_moss.jpg) |
| no crack | lumpy surface| crack in large context |
| ![](/assets/images/crack/noncrack.jpg) | ![](/assets/images/crack/crack_noise.jpg)| ![](/assets/images/crack/crack_in_large_context.jpeg)|


I am very welcome to further idea from you. please drop me an email at khanhhh89@gmail.com if you think of other cases

# Training
- step 1. download the dataset from [the link](https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP)
- step 2. run the training code
- step 3:
```python
python train_unet.py -data_dir PATH_TO_THE_DATASET_FOLDER -model_dir PATH_TO_MODEL_DIRECTORY -model_type resnet_101
```

# Result
The best result is achieved by UNet_Resnet_101 with IoU = and Dice =

| Model            | IOU | Dice |  
|------------------|---------|---------|
| UNet_VGG16       | mean = 0.4687, std = 0.2217  | mean = 0.6033, std = 0.2382|
| UNet_Resnet_101 | mean = 0.3861, 0.2123  | mean = 0.51877, std = 0.2538  |
| DenseNet         |       |        |
