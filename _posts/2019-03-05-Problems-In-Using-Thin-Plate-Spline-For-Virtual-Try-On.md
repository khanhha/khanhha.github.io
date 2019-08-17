---
title: "Problems in  using Thin-Plate-Spline for image-based virtual try on"
categories:
  - posts
tags:
  - pytorch
  - deep_learning
  - virtual_try_on
  - image_deformation
---


In this note, I describe two problems about the Geometric Matching Module in [the paper](https://arxiv.org/pdf/1807.07688.pdf): "Toward Characteristic-Preserving Image-based Virtual Try-On Network" that I found out during my research. The note requires the background knowledge about the virtual try on method, which are described in detail in the the paper or my previous articles [1](https://khanhha.github.io/posts/Image-based-Virtual-Try-On-Network-Part-1/) and [2]().

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Incorrect target clothes mask.](#incorrect-target-clothes-mask)
  - [Description](#description)
  - [Idea](#idea)
- [TPS fails to handle self-intersection.](#tps-fails-to-handle-self-intersection)
  - [Description.](#description-1)
  - [Idea](#idea-1)
- [Conclusion](#conclusion)

<!-- /code_chunk_output -->

# Incorrect target clothes mask.

## Description

As stated in the paper, the training data for the Geometric Matching Module consists of a person representation $p$, an in-shop clothes image, and the corresponding clothes region on the human subject, as depicted in below.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-097741a9.png" width="600px" />
    <div class="caption">images are taken from the paper</div>
</div>
<br/>

The clothes regions on the human subject are marked by the green contour as in the below figure. From the implementation by the author, these green contours define the clothes regions to which the in-shop clothes on the left is warped.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-39edeab0.png" width="700px" />
    <div class="caption">occluded regions by arms</div>
</div>
<br/>

When the clothes regions are completely visible, a good mapping could be found because the percentage of 1-1 pixel correspondence is high. However, when the clothes are occluded by hair and arms, 1-1 mapping does not exist for many pixels, which causes problem for the TPS transformation. The model is not able to learn to ignore the occluded clothes region, and therefore, produces strange result as follows.


<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-7fc9b559.png" width="500px" />
    <div class="caption">bad warping result due to occlusion</div>
</div>
<br/>

In addition to occlusions caused by hair and arms as above, too different view and tuck also cause clothes to be invisible.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-2ef8f008.png" width="500px" />
    <div class="caption">bad warping result due to occlusion</div>
</div>
<br/>

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-0d2b2f17.png" width="500px" />
    <div class="caption">bad warping result due to occlusion</div>
</div>
<br/>

## Idea
One idea to solve this problem is training an  in-painting clothes model to fill in occluded regions by air and arm. The in-painted clothes will be then used as the training target for the Geometric Matching Module. Basically, the result from the in-paining module will be similar my below manual modification.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-19b4443f.png" width="500px" />
    <div class="caption">clothes inpainting</div>
</div>
<br/>

# TPS fails to handle self-intersection.

## Description.
Self-intersection occurs with long-sleeve shirts, when the human subject put their arms in front of their belly. In this case, the Thin Plate Spline deformation fails to handle the sleeve correctly. My below experiment demonstrate this problem. First a set of correspondences are specified about the torso contour of the clothes. In this case, no self-intersection among the control point occurs; therefore, the TPS is able to warp the torso region of the in-shop clothes to align with the corresponding torso region of the clothes target.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-fc563f97.png" width="500px" />
    <div class="caption">image is modified from the lecture note by Frédo Durand</div>
</div>
<br/>

The warping result with the above point correspondences is shown in below.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-ebfe54c6.png" width="500px" />
    <div class="caption"></div>
</div>
<br/>

However, when more control points along the sleeve are specified, the TPS starts to show its inability to handle self-intersection.
<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-233f9bc1.png" width="500px" />
    <div class="caption"></div>
</div>
<br/>

And below is the warping result with the added control points.


<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-3dd0b548.png" width="500px" />
    <div class="caption"></div>
</div>
<br/>

## Idea
To deal with this problem, other image deformations such as bounded biharmonic deformation could be used, as show in the below picture. However, the problem is that the chosen deformation method needs to be able to integrate into the training process. This seems very challenging to me now becaues the biharmonic deformation requires to triangulate the clothes contour, and the triangulation could change from contour to contour.

<div style="align: left; text-align:center;">
    <img src="/assets/images/vton/2019-7287ca6b.png" width="500px" />
    <div class="caption"></div>
</div>
<br/>

Another approach is apply TPS for each clothes part such as upper sleeve, lower sleeve, and torso independently. Unfortunately, some kind of clothes do not have sleeves, so how can we integrate this knowledge into the training a neural network? I am still trying to get my head around it.

# Conclusion
Further problems will be updated along my research.
