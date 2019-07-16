---
title: "3D Human Dataset"
categories:
  - posts
tags:
  - human_estimation, human_dataset
---

In this note, I summerised different human data-sets that I collected so far for my research. For each dataset, you can find the link to the dataset homepage, the main data types provided by the dataset and its sizes. I hope that it would be helpful for you as well.

# [Caesar dataset](http://store.sae.org/caesar/)
![](/assets/images/3dhmds/caesar.png){:width='500px'}
- Keywords: 3D_shape, single_pose
-  The original Caesar data-set contains 3D scan meshes of 2,400 U.S. & Canadian and 2,000 European civilians, half of which are female. Each mesh comes with 72 body landmarks as shown in the above figure.

- [The Maxplank Institute version](http://humanshape.mpi-inf.mpg.de)
  - Data: Caesar meshes are registered to a template triangle mesh of 6890 vertices and 12K triangles. There are no raw scans, heights, weights or gender come with the published dataset from the website.
  - Size: Around 4300 registered meshes.

- [The UCSC version](https://graphics.soe.ucsc.edu/data/BodyModels/index.html)
  - Data: male and female meshes are registered to a common topology of 12500 vertices and 25000 facets. The raw scans, weights and heights are not available.
  - Size: 1500 registered males and 1500 registered females.

- Note: I am not sure if meshes from the UCSV and from The Maxplank institute are overlapping or not for two reasons. The file names of the meshes are completely different, and the version from UCSC just contains 3000 meshes, while there are 4300 around meshes from the Maxplank's dataset.

# [Scape Dataset](http://ai.stanford.edu/~drago/Projects/scape/scape.html)
![](/assets/images/3dhmds/scape.png)
- Keywords: 3D_shapes, static_poses
- Data: 71 meshes of a particular person in different poses registered to the same topology. The original meshes consist of 125K polygons and the registered mesh has 25K polygons.


# [Scan DB](http://gvvperfcapeva.mpi-inf.mpg.de/public/ScanDB/)
![](/assets/images/3dhmds/scandb.png)
- Keywords: 3D_shapes, static_poses
- Data: 550 full body 3D scans of 114 subjects, each of which were scanned in at least 9 poses sampled randomly from 34 poses. The captured scans are brought to the same template mesh using non-rigid registration.

# [Faust Dataset](http://faust.is.tue.mpg.de/overview)
![](/assets/images/3dhmds/faust.png)
- Keywords: 3D_shapes, static_poses
- Data: 300 triangle meshes of different subjects scanned in 30 different poses. All the original scanned mesh are registered to the same topology of 6890 vertices. Each mesh comes with 17 body landmarks.

# [D-Faust Dataset](http://dfaust.is.tue.mpg.de/)
![](/assets/images/3dhmds/dfaust.png)
- Keywords: 3D_shapes, dynamic_poses
- Data:  The data-set contains 3D scans of 10 non-rigid human subjects captured at 60 fps, which results in around 40.000 raw meshes. All the meshes are aligned to the same template mesh.

# [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
![](/assets/images/3dhmds/human360m.png)
- Keywords: 3D_shapes, dynamic_poses
- Data: the data-set contains 3.6 million different human poses of 11 professional actors (6 male and 5 female) taken from 4 digital cameras. The dataset come with the following data
  - 3D pose is given with respect to a skeleton.
  - time-of-flight data
  - 3D scanning meshes of actors


# [HumanEva](http://humaneva.is.tue.mpg.de/)
![](/assets/images/3dhmds/humaneva.png)
- Keywords: dynamic_poses
- Data: 3D body poses of 4 human subjects performing 6 common actions (walking, jogging, gesturing, etc.) The datset contains 7 calibrated video sequences (4 gray and 3 color).

#[UP-3D dataset](http://files.is.tuebingen.mpg.de/classner/up/)
![](/assets/images/3dhmds/up.png)
- Keywords: RGB_images, SMPL_params
- Data:  The UP-3D dataset consists of 26,239 images with SMPL parameters merged from 3 human datasets: [ Leads Sports Pose](http://www.comp.leeds.ac.uk/mat4saj/lsp.html), [MPII Human Pose](http://human-pose.mpi-inf.mpg.de/) and FashionPose.
The dataset is created by fitting the SMPLify model parameters to real human images in a way that the reprojection of estimated 3D human model match with the silhouette of the human subject in the image. The bad fittings are then sorted out by human annotators.


# [SURREAL Dataset](https://www.di.ens.fr/willow/research/surreal/)
![](/assets/images/3dhmds/surreal.png)
- Keywords: synthetic_RGB_images, SMPL_params
- This synthetic dataset is created by rendering random human subjects represented by the SMPL model from random perspectives, lighting and backgrounds. The SMPL shape parameters are sampled randomly from the Caesar dataset. The SMPL pose parameters are created by fitting motion capture skeleton data from the dataset [CMUMoCap](http://mocap.cs.cmu.edu/) to the SMPL model. Regarding subject appearance, 3D body are registered with human textures extracted from real human scans and illuminated using random Spherical Harmonics coefficients.

- Data: human parts segmentation, pose, depth, SMPL parameters.

- Size: 6 million frames synthesized from 400K background images, 4300 real human shapes and 2000 sequences of 23 high-level action categories.

# [People Snapshot Dataset](https://graphics.tu-bs.de/people-snapshot)
![](/assets/images/3dhmds/snapshot.png)
- Keywords: 3D_shapes, 3D_shapes_in_clothes, static_pose
- Data: Short clips of male/female subjects rotating in front of a camera, 3D obj files with texture, 3D keypoints, masks and poses.
- Size: 24 human subjects in casual, sport or plaza clothes.
- The dataset is created by having subjects rotating in front a camera, which produces an image sequence with corresponding segmentations. From these data, the silhouette camera rays are estimated to optimize for the subjects shape in T-Pose. Therefore, the obj files in the datasets are not the groundtruth labels of the subjects.

# [BUFF - Bodies Under Flowing Fashion](http://buff.is.tue.mpg.de/)
![](/assets/images/3dhmds/buff.png)
- Keywords: 3D_shapes, 3D_shapes_in_clothes, dynamic_poses
- Data: 3D scans and 3D scan sequences of people under clothes, the corresponding naked shapes in the static pose, per frame poses, per frame detailed template shapes.
- Size: 3 males and 3 females, 2 clothing styles, 3 motions totalling 13,600 scans.
- The dataset is created by using a custom-built multi-camera active stereo system to capture temporal sequence of full-3D scans, which are then fed to algorithms to estimate the naked shapes under clothes. Because the naked 3D shapes are calculated from the raw scans, they can only considered as pseudo groundtruth for human shape prediction problems.
