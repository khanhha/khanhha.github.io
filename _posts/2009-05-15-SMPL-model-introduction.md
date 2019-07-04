---
title: "SMPL human model introduction"
date: 2019-04-18T15:34:30-04:00
categories:
  - posts
tags:
  - human_estimation
---

# Shape synthesis

The rest pose shape is formed by a linear combination of principal shape components, which are vertex deviations from the mean shape of the data-set. Specifically, each principal component is a vector of $6890\text{x}3$, which represent $(x,y,z)$ displacements from the corresponding vertices of the mean mesh. To make it clear, below is a visualization of the first and second principal components of the SMPL model, which denotes the largest changes among the meshes in the data-set. The mesh pair for each component is computed as follows:
$$ M_i = T \pm 3{\sigma}*PC_k$$

Intuitively, it seems that the first component explains for the change in height and the second represents the change in weight among human meshes.

![](/assets/images/smpl/pca_1_2.png)

After the model is trained, a new shape could be created by linearly combining 10 principal components from the SMPL model, as shown in the below code.
```python
# shapedirs:  6890x3x10:  10 principal deviations
# beta:       10x1:       the shape vector of a particular human subject
# template:   6890x3:     the average mesh from the dataset
# v_shape:    6890x3:     the shape in vertex format corresponding to the shape vector
v_shaped = self.shapedirs.dot(self.beta) + self.v_template
```

# Joint locations
In this step, 24 skeleton joint locations are estimated from the mesh calculated in the previous step. The idea is to build a set of sparse neighbor vertices for each joint, and then the joint location for another mesh could be calculated  as the weighted average of the neighbor vertex set. This average is represented by a joint regression matrix learned from the data-set that defines a sparse set of weight for each joint.

![](/assets/images/smpl/joint.png)

```
# v_shape:          6890x3    the mesh in neutral T-pose calculated from a shape parameter of 10 scalar values.
# self.J_regressor: 24x6890   the regression matrix that maps 6890 vertex to 24 joint locations
# self.J:           24x3      24 joint (x,y,z) locations
self.J = self.J_regressor.dot(v_shaped)
```

# Pose Blend Shapes
In the SMPL model, the human skeleton is described by a hierarchy of 24 joints corresponding to the white dots in the below figure. This hierarchy is defined by a kinematic tree that keeps the parent relation for each joint.
![](/assets/images/smpl/joint_locations.png)

These 24 joints form the pose parameter of the model, which is vector of $(23\text{x}3)$ representing $23$ (except the origin joint) relative rotations of $23$ joints from their parent joints. Each rotation is encoded by a axis-angle rotation representation of $3$ scalar values, which is denoted by the $\boldsymbol{\theta}$ vector in the below figure.
![](/assets/images/smpl/axis_angle_rot.png)

The relative rotations of 23 joints $(23\text{x}3)$ causes deformation to surrounding vertices. These deformations are captured by a matrix of (23x6890x3) which represents $23$ principal components of vertex displacements of $(6890x3)$. Therefore, given a new pose vector of 23 values, the final deformation will be calculated as a linear combination of these principal components.

```python
# self.pose :   24x3    the pose parameter of the human subject
# self.R    :   24x3x3  the rotation matrices calculated from the pose parameter
pose_cube = self.pose.reshape((-1, 1, 3))
self.R = self.rodrigues(pose_cube)

# I_cube    :   23x3x3  the rotation matrices of the rest pose
# lrotmin   :   207x1   the relative rotation values between the current pose and the rest pose   
I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
lrotmin = (self.R[1:] - I_cube).ravel()

# v_posed   :   6890x3  the blended deformation calculated from the
v_posed = v_shaped + self.posedirs.dot(lrotmin)
```
