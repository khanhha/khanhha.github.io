---
title: "SMPL human model introduction"
date: 2019-04-18T15:34:30-04:00
categories:
  - posts
tags:
  - human_estimation
---

# Introduction
This article could be served as a bridge between the theoretical explanation in SMPL paper and a sample numpy code that synthesizes a new human subject from a pre-trained model provided by the Maxplank Institute. I wrote it as an exercise to strengthen my knowledge about the implementation of the SMPL model.  

3D objects are often represented by vertices and triangles that encodes their 3D shape. The more detail an object is, the more vertices it is required. However, for objects like human, the 3D mesh representation could be compressed down to a lower dimensional space whose axes are like their height, fatness, bust circumference, belly size, pose etc. This representation is often smaller and more meaningful.

The SMPL is a statistical model that encodes the human subjects with two types of parameters:
- Shape parameter: a shape vector of 10 scalar values, each of which could be interpreted as an amount of expansion/shrink of a human subject along some direction such as taller or shorter.

- Pose parameter: a pose vector of 24x3 scalar values that keeps the relative rotations of joints with respective to their parameters. Each rotation is encoded as a arbitrary 3D vector in axis-angle rotation representation.

As an example, the below code samples a random human subject with shape and pose parameters. The shift and multiplication are applied to bring the random values to the normal range of the SMPL model; otherwise, the synthesized mesh will look like an alien. 
```python
pose = (np.random.rand((24, 3)) - 0.5)
beta = (np.random.rand((10,)) - 0.5) * 2.5
```
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

The relative rotations of 23 joints $(23\text{x}3)$ causes deformation to surrounding vertices. These deformations are captured by a matrix of (23x6890x3) which represents $23$ principal components of vertex displacements of $(6890x3)$. Therefore, given a new pose vector of relative rotation 23x3x3 values as weights, the final deformation will be calculated as a linear combination of these principal components.

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

# Skinning
when the joints transforms, the neighboring vertices are also transformed with the same transformations. The further a vertex from a joint, the less it is affected by the joint transformation. Specifically, a final vertex is calculated as a weighted average of 24 transformed versions, corresponding to 24 joints.
![](/assets/images/smpl/skinning.png)
The below code first calculates the global transformation for each joint by recursively concatenating its local matrix with its parent matrix. These global transformations are then subtracted from the corresponding transformations of the joints in the rest pose. For each vertex, its final transformation is calculated by blending the 24 global transformations with different weights. The code for these steps are shown in the below.

```python
# world transformation of each joint
G = np.empty((self.kintree_table.shape[1], 4, 4))
# the root transformation: rotation | the root joint location
G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
# recursively chain transformations
for i in range(1, self.kintree_table.shape[1]):
  G[i] = G[self.parent[i]].dot(
    self.with_zeros(
      np.hstack(
        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
      )
    )
  )

# remove the transformation due to the rest pose
G = G - self.pack(
  np.matmul(
    G,
    np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
    )
  )
# G       : (24, 4, 4)  : the global joint transformations with rest transformations removed
# weights : (6890, 24)  : the transformation weights for each joint
# T       : (6890, 4, 4): the final transformation for each joint
T = np.tensordot(self.weights, G, axes=[[1], [0]])

# apply transformation to each vertex
rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

# add with one global translation
verts = v + self.trans.reshape([1, 3])
```
