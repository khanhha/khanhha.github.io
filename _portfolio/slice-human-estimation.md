---
title: Slice-Based Human Reconstruction
excerpt: A full pipeline that reconstructs a 3D human model based on a combination of different machine learning techniques and geometry processing.
header:
  teaser: /assets/images/human/human_slice_measure.jpg
---


Because this is a commercial project, I am not allowed to provide source code or explain in detail about the steps of the pipeline. I can only give you a few words about the pipeline. Hopefully I will give you a general idea about what I worked on in this project.

This project is about reconstructing a human 3D model by cutting the front and side silhouette into slices and then, use the slice information as the information to machine learning models such as decision tree to predict a control mesh that define a coarse shape of the human model. This coarse shape will be then used to generate the final dense mesh of the human.

The implementation of stages in the method requires the usage of different machine learning.
- In the first stage of segmenting the silhouette from the front/side images, the traditional methods like grab-cut doesn't result in good contour under noisy condition, so a deep learning approach is used instead. It still requires some post-processing steps, but in general, the result is much better than the traditional methods.
- Other deep learning models are also used to extract landmark locations on human body for a better slice location identification. The below figure depicts the result of the first step.

  ![human_slice](/assets/images/human/human_slice_measure.jpg){: height='350px' .center-image}

- Machine learning models like decision tree are used to predict the shapes of slices of human body. These slices will be put together to form a coarse mesh that present a coarse version of the human. The final result is similar to the below figure.
  ![human_ctl_mesh](/assets/images/human/control_mesh.jpg){: height='350px' .center-image}

Finally, regarding the learning aspect, this project gave me a lot of opportunities to solve real challenges which often come up in industrial projects, but unfortunately are not covered in academic papers. In addition to machine learning techniques, the project also involves different image processing and geometry processing techniques, which really help diverse my skill sets and increase my flexibility in choosing an appropriate set of techniques to solve a real-world problem.


For further questions, you can send me an email at khanhhh89@gmail.com.
Thanks for your time reading this
