---
title: "End To End Human Recovery Based On Generator-Discriminator Machanism"
categories:
  - posts
tags:
  - human_estimation
  - deep_learning
---

This is a note that I jotted down while reading the paper "End-to-end Recovery of Human Shape and Pose" as a way to deepen and reorganize my knowledge. It could be also seen as a bridge between mathematical equations in the paper and the corresponding Pytorch code. I wrote it mainly for myself, but I also hope that it would be useful for you to navigate through the paper and the Pytorch code.

The implementation was written by MandyMo. You can check it out [here](https://github.com/MandyMo/pytorch_HMR). I chose this version because it seems more understandable to me than the original Tensorflow version by the author.

- [Datasets](#datasets)
- [Overview](#overview)
- [Generator](#generator)
- [Generator Losses](#generator-losses)
- [Discriminator](#discriminator)
- [Discriminator Losses](#discriminator-losses)

# Datasets
## 2D datasets
- [LSP and LSP-extended dataset](http://sam.johnson.io/research/lsp.html): 12K images, 14 2D keypoints per image
- [MPII human pose dataset](http://human-pose.mpi-inf.mpg.de/): 25K images, 14 2D keypoints per image
- [MS-COCO dataset](http://cocodataset.org/#overview): 200K images, 2D keypoints per image.
## 3D dataset
- [Human3.6M](http://vision.imar.ro/human3.6m): 3.6 million images/video frames of 11 actors, 2D keypoints per image, 3D keypoints per image, SMPL parameters.
- [MPIINF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/): 8 subjects, covering 8 activities with 2 sequences per subject, 4 minintes per sequence, 2D keypoints per image, 3D keypoints per image, camera parameters.  
- [Pose Prior dataset](http://poseprior.is.tue.mpg.de/): images, 2D keypoints, 3D keypoints
- [CMU pose dataset](http://mocap.cs.cmu.edu/): images, 2D keypoints, 3D keypoints

<br/>
<br/>
# Overview
<div style="align: left; text-align:center;">
    <img src="/assets/images/e2ehm/overview.png" width="700px" />
    <div class="caption">image is taken from the paper</div>
</div>
<br/>

The general pipeline of the end-to-end model is depicted in the above figure.

As shown in the green stage, the generator module takes in an RGB image and spits out 85 scalar values of pose $\theta$, shape $\beta$, and camera parameters $\{s,R,T \}$ that describe the human subject and its projection on the image.

In the next red stage, the shape and pose parameters: $\theta$ and $\beta$ are passed into the SMPL (a human statistical model) to reconstruct the full 3D mesh of the human subject. This 3D mesh also gives us the 3D joint locations.

The pink stage means that the predicted 3D joint locations are projected to the image using camera parameters. The reprojected 2D keypoints will be used to compare against the ground truth 2D keypoints to optimize the generator.

Finally, in the blue stage, a discriminator is trained to encourage the generator to spit out reasonable values that represents a real human subject; otherwise, as stated in the paper, the generator will produce many visually displeasing result. To achieve it, the generator's weights will be trained, through the discriminator, to fool the discriminator to believe that the $\Theta$ vectors from the generator are from the dataset. At the same time, the discriminator is also fed with both fake values from generator and ground truth values from the dataset to better recognize which one comes from the dataset and which one does not. Over time, the discriminator will become more precise, which also pushes the generator to be more delicate to be able to fool the discriminator.


<br/>
<br/>
# Generator
<div style="align: left; text-align:center;">
    <img src="/assets/images/e2ehm/generator.png" width="700px" />
    <div class="caption">image is taken from the paper</div>
</div>
<br/>


The HMR paper represents a human subject as an 85-dimensional vector
$\Theta = \{\theta, \beta, R, t, s \}$
- $\theta$ and  $\beta$ are shape and pose parameters to the SMPL model. Given these two parameter vectors as input, the SMPL model will return a posed mesh of the human subject.  
- $R, t, s$ represents rotation, translation, and scale of the camera.

As shown in the above figure, the generator consists of two main blocks: encoder and regressor. The encoder takes in an RGB image and produces image features, which could be a vector of 4096, 2048 or 1664 values depending on encoder types such as Resnet50, Densenet or Hourglass, etc. These image features are then passed to the Regressor to predict the $\Theta$ vector that represents the subject.

The code for the whole generator is shown below. It first forwards the input image to the encoder and then passes that feature next to the regressor to predict the $\Theta$ vector.

__Generator__
```python
class HMRNetBase(nn.Module):
    def forward(self, inputs):
        if self.encoder_name == 'resnet50':
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name.startswith('densenet'):
          pass #Removed for convenience
        elif self.encoder_name == 'hourglass':
          pass #Removed for convenience

```
__Regressor__
```python
class ThetaRegressor(LinearModel):
    '''
    param:
        inputs: is the output of encoder, which has 2048 features
    return:
        a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations X N X 85(or other theta count)
    '''
    def forward(self, inputs):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)
        return thetas
```

As mentioned in section 3.2 of the paper, it is challenging to regress the $\Theta$ vector in one forward step, especially for the rotation part. Therefore, the paper employs a self-correcting approach that iteratively adjusts an initial solution by feeding back error predictions. Instead of predicting the $\Theta$ vector directly, the regressor starts with the mean theta vector and returns how far the input vector is from the true value. This correction vector is then added back to the input to create the input for the next iteration.

After $N$ iterations, we have a set of prediction values $(\Theta_0,..., \Theta_n)$, which will be used later to calculate loss values. The whole process is summarized in the method $forward$ of the $ThetaRegressor$. For more detail about the Iterative Error Feedback, please refer to [the paper](https://arxiv.org/pdf/1507.06550.pdf)

<br/>
<br/>
# Generator Losses
Now we will discuss different losses for optimizing the generator. There are 3 main types of loss: 2D keypoint loss, 3D keypoint loss, and SMPL parameters loss. The two later losses are just calculated over the two 3D datasets whose 3D ground truth data is available.

## 2D Keypoint Loss
The 2D keypoint loss is calculated as L1 distance between the reprojected keypoints and the ground truth keypoints. Specifically, the regressor predicts SMPL parameters, which are  used to reconstruct the posed 3D mesh of the human subject. Thanks to the fixed topology of the mesh, 3D joint locations could be inferred using adjacent vertices. The estimated 3D joint locations are then projected to 2D keypoints using camera parameters $\{s, T, R\}$. The below implementation is a realization of the loss equation $(3)$ in the paper, as shown in below

$$
L_{reproj} = \sum_i||v_i(x_i - \hat{x_i})||_1
$$

where $x_i$ are 2D keypoints, $\hat{x_i}$ are reprojected 2D keypoints, $v_i\in(0,1)$  is the visibility of each keypoint

```python
"""
    purpose:
        calc L1 error
    Inputs:
        kp_gt  : N x K x 3
        kp_pred: N x K x 2
"""
def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predict_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k
```
## 3D Keypoint Loss
The 3D keypoint loss is calculated as L2 distance between the predicted 3D keypoints and the ground truth keypoints. To avoid the difference in global translation, the two keypoint set are first transformed to its local space by subtracting every keypoints from the average point of the left and right pelvis, as done by the function __align_by_pelvis__. Then the loss is defined as the mean of squared distances of K keypoint pairs.

Note that the 3D keypoint loss is just calculated for data records where the 3D ground truth keypoints are available.

The loss is described by the equation $(6)$ in the paper, as rewritten as below

$$
L_{joints} = ||(X_i - \hat{X_i})||_2^2
$$

```python
'''
    purpose:
        calc mse * 0.5
    Inputs:
        real_3d_kp  : N x k x 3
        fake_3d_kp  : N x k x 3
        w_3d        : N x 1
'''
def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
    shape = real_3d_kp.shape
    k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8

    #first align it
    real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
    kp_gt = real_3d_kp
    kp_pred = fake_3d_kp
    kp_dif = (kp_gt - kp_pred) ** 2
    return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
```

## SMPL shape and pose loss
When the ground truth SMPL shape and pose parameters are available, the SMPL loss is calculated as the L2 distance between predicted parameters and ground truth parameters.

In contrast to the shape loss which is drawn directly on shape parameters, the pose loss is calculated based on the rotation matrix representation, which is converted from the corresponding angle-axis representation using [Rodrigues formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula). This implementation is different from the equation in the paper as follows:

$$
L_{smpl} = ||[\beta_i, \theta_i] - [\hat{\beta_i},\hat{\theta_i}]||_2^2
$$


```python
'''
    purpose:
        calc mse * 0.5
    Inputs:
        real_shape  :   N x 10
        fake_shape  :   N x 10
        w_shape     :   N x 1
'''
def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
    k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
    shape_dif = (real_shape - fake_shape) ** 2
    return  torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k
'''
    Input:
        real_pose   : N x 72
        fake_pose   : N x 72
'''
def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
    k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
    real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:], batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:]
    dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
    return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k
```
<br/>
<br/>
# Discriminator
<div style="align: left; text-align:center;">
    <img src="/assets/images/e2ehm/discriminator.png" width="700px" />
    <div class="caption">image is taken from the paper</div>
</div>
<br/>

As explained in the section $3.3$ of the paper, the reprojection loss just ensures that the difference between reprojected keypoints and ground truth keypoints is minimized, even that the 3D configuration is irregular because one 2D configuration could be explained by multiple 3D configurations. It brings the need for a discriminator that is trained to encourage the Generator to churn out reasonable SMPL parameters.

According to the paper, there are 3 types of discriminators: shape discriminator, pose discriminator for each joint rotation, and pose discriminator for the whole SMPL pose parameter. Factorizing discriminators this way help reduce visually displeasing results returned by the Generator.

## The shape discriminator
From 10 SMPL parameter, the shape discriminator predicts the probability that the input comes from the real data.
```python
'''
    shape discriminator is used for shape discriminator
    the inputs if N x 10
'''
class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)
        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)
```

## The Pose Discriminator
The pose discriminator includes two main blocks. The convolutional block __self.conv_blocks__ applies convolutional layers over the input pose data of size $9\times1\times23$ and outputs an intermediate data of $c\times1\times23$ where c is the number of channels. After that, the 23 pose discriminators, specialized for 23 joints, will apply another linear layer over each data slice of $1\times23$ to produce the probability if a joint angle is within the normal limits.

```python
class PoseDiscriminator(nn.Module):
    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(channels[-1])
            sys.exit(msg)

        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name = 'conv_{}'.format(idx),
                module = nn.Conv2d(in_channels = channels[idx], out_channels = channels[idx + 1], kernel_size = 1, stride = 1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features = channels[l - 2], out_features = 1))

    # N x 23 x 9
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2) # to N x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs) # to N x c x 1 x 23
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:,:,0,idx]))

        return torch.cat(o, 1), internal_outputs
```

## The Full Pose Discriminator
The full discriminator takes in the intermediate data returned by the previous Pose Discriminator and returns the probability if a pose parameter set comes from the data.
```python
class FullPoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(FullPoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)
```

## Final Discriminator
The final discriminator that groups the shape, specific pose, full pose discriminators in a single block is shown in the below code. It first decomposes the $\Theta$ values returned by the Generator into pose and shape parameters, which are then passed to separated discriminators. The output of the __forward__ method is 25 probability values: 23 for 23 joints, 1 for the full pose parameters and another 1  for shape parameters.
```python
class Discriminator(nn.Module):
  '''
    inputs is N x 85(3 + 72 + 10)
  '''
  def forward(self, thetas):
      batch_size = thetas.shape[0]
      cams, poses, shapes = thetas[:, :3], thetas[:, 3:75], thetas[:, 75:]
      shape_disc_value = self.shape_discriminator(shapes)
      rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
      pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
      full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
      return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)
```
<br/>
<br/>
# Discriminator Losses
As described in the section 3.3 of the paper, there are two types of discriminator losses:  the adversarial loss function for the encoder and the loss for each discriminator.
## The adversarial loss function
The adversarial loss function as rewritten in the below, tries to optimize the encoder (generator) weights, through the discriminator, in a way that the generator will produce shape and pose parameters that could fool the discriminator to believe that they are true parameters. In other words, each probability $D_i$ is subjected from $1$ means that the generator should be trained in a way that its shape, pose output will make the discriminator $D_i$ return a probability as close to $1$ as possible.

$$
L_{adv}(E) = \sum_i\mathbb{E}_{\Theta\backsim{pe}}[(D_i(E(I)) - 1)^2]
$$  

The code for the loss function is shown in below. The input is 25 probability values, as explained in the previous paragraph.
```python
'''
  Inputs:
      disc_value: N x 25
'''
def batch_encoder_disc_l2_loss(self, disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k
```
## The objective loss for each discriminator
The objective loss for each discriminator consists of two terms, as shown in the below equation. The first term means that given the ground truth $\Theta$ value, the discriminator should return a probability value close to $1$. In contrast, the second term tell discriminator that given the predicted $\Theta$ from the generator, its output probability should be as small as possible.

$$
L(D_i) = \mathbb{E}_{\Theta\backsim{p_{data}}}[(D_i(\Theta) - 1)^2] +         \mathbb{E}_{\Theta\backsim{p_{E}}}[D_i(E(I))^2]
$$  

One very important point we need to notice here is that optimizing this loss doesn't result in a change in weights of the generator. Here we just optimize the discriminator weights, which is in contrast to the previous loss, whose optimization would cause the change in weights of the generator.
The implementation of the loss is shown below.
```python
'''
    Inputs:
        disc_value: N x 25
'''
def batch_adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb
```
