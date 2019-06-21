---
title: A graphics module for the 3D printer software Nyomo
excerpt: An OpenGL-based graphics engine that supports visualization features for the 3D printer software Nyomo
header:
  teaser: /assets/images/nyomo/nyomo_demo.jpg
sidebar:
  - title: "Role"
    #image: http://placehold.it/350x250
    #image_alt: "logo"
    text: "Senior Developer"
  - title: "Responsibilities"
    text: "Develop the graphics engine and visualization features"
---

During my time at [Nyomo](https://www.youtube.com/channel/UChObDqXzc4RD5gLZmF_Ey3w), a 3D printer software that was based in Singapore, I developed the graphics module for the software that visualize the 3D objects to user in a way that make it easier for users to distinguish the small details and curvature of 3D object surface. The graphics module that I wrote is also used to visualize other functionalities of the software. Below is a quick tour through the main features of the software.

- Curvature-based rendering
![mesh_curvatuve](/assets/images/nyomo/mesh_curvature.jpg){:height='300px'}
In this feature, the surface of the 3D object is visualized in a way that the curved-down regions like the blue ones are darker than the curved-up regions like the green one. Through the feedback of the customers, this feature really helped enhance the visual quality of the software and make the experience with the software more interesting.
The feature was implemented based on the algorithm ["Radiance Scaling for Versatile Surface Enhancement"](https://hal.inria.fr/inria-00449828/file/RadianceScaling.pdf)

- Hovering object highlighting
![hovering_effect](/assets/images/nyomo/mesh_hovering.jpg){:height='300px'}
When the number of 3D objects is dense, there is a need for highlighting the object being under the mouse pointer, so that it will look distinguished from the surrounding objects. Normally, 3D software implements this effect by drawing a bright outline along the silhouette of the object. However, this approach requires two rendering steps, which reduces the FPS of the software and it also looks kind of unimpressive. In our 3D printer software, an highlighting mechanism is implemented so that the hovering object will be colored different at regions where vertex normal vectors are perpendicular to the viewing vector.

As you see in the above picture, pixels inside the blue-contoured regions are mixed with more blue color than other regions. The mixing ratio is calculated based on the angle between the blue vertex normal vectors and the viewing screen vector, which is point inside the screen. I am very proud of this simple trick because it make the user experience with the software much more amazing.

## Contact
if you have further question about the features, you can contact me through my email khanhhh89@gmail.com
