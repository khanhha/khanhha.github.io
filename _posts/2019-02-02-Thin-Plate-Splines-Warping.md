---
title: "Thin Plate Splines Warping"
categories:
  - posts
tags:
  - warping
  - computational_photography
---

# Introduction
I have been using the Thin Plate Warping algorithm from OpenCV for quite a long time, but still just have a vague idea about it.  It often makes me feel uncomfortable. Therefore, I set out to read the paper about Thin Plate Warping and dig deep in to the OpenCV implementation to get a better insight about the method.  This article is my summary I wrote along my journey, as a way to reorganize concepts. Hopefully it would be useful for your as it was to me.

# What is an image warping problem?
Given an image with a sparse set of control points $(x_i, y_i)$ with corresponding displacements $(\Delta{x_i}, \Delta{y_i})$ , we want to find a mapping $f: (x,y) \to (x', y')$ from pixels in the input image to pixels in the warped/deformed image so that the corresponding warped control points $(x'_i, y'_i)$ closely match its expected targets $(x_i + \Delta{x_i}, y_i + \Delta{y_i})$.  

As shown in the below figure, the red arrows represent $4$ point correspondences between the left and the right image. The origin of red arrows represent control points $(x_i,y_i)$ in the left image and the end of the arrows represent the locations of the corresponding control points $(x_i+\Delta{x_i}, y_i+\Delta{y_i})$ in the right image. Given this information, we want to find a smooth interpolation function or a warping field that maps all 2D points from the left image to the right image. The result of the warping algorithm will look like the right below image.

![](/assets/images/tps/2019-10d65de0.png)

In the 1-dimensional space, this problem could be stated that given a sparse set of correspondences $(x_i, x'_i)$ as shown by red points, we want to estimate a smooth function that maps every points $x$ to $x'$

![](/assets/images/tps/2019-bb55c491.png)

# Image Warping Applications

Further examples of image warping is shown in below.

## Image editting
![](/assets/images/tps/2019-d9d9c494.png)

## Texture mapping
Given two set of facial landmarks in the input image and in the texture space, we can warp the input image to the texture so that it could be displayed in 3D.
![](/assets/images/tps/2019-18845bd0.png)

## Image morphing
Given two set of facial landmarks in two images, one face could be warped to make it look like the other face.
![](/assets/images/tps/2019-17b79ccd.png)

## Radial Basis Function interpolation
Before introducing the Thin Plate Spline warping algorithm, we will quickly go through the radial basis function interpolation, which is the general form of the thin plate spline interpolation problem. We will just focus on explaining the $1D$ example in the last paragraph.

In the beginning, the only information we have is a sparse set of control point correspondences. The most simple way to infer the other points is linear interpolation, where interpolated points will move along the segment connecting the two closest control points, as shown in below.
![](/assets/images/tps/2019-cd769232.png)

However, what we want is a smooth interpolation function that goes through control points closely, which is where Radial Basis Function comes to help. As explained in [this lecture note](http://groups.csail.mit.edu/graphics/classes/CompPhoto06/html/lecturenotes/14_WarpMorph.pdf), Radial basis functions are basically smooth kernels centered around control points $x_i$. The further an interpolated point $x$ from a control point $x_i$, the smaller its distance to the corresponding kernel and therefore, the less it is affected by the kernel.

![](/assets/images/tps/2019-fe1ff27f.png)

In general, instead of interpolating from just two closest points as in our above linear interpolation example, with radial basis function, each point will be calculated a weighted combination of all kernel distances, as summarized by the following function. Given an input point $x$, this smooth interpolation function will return the corresponding warped point $x' = f(x)$
$$
f(x) = \sum{\alpha_iR(x, x_i)}
$$
where $\alpha_i$ is the weight corresponding to the radial basis function $R(x,x_i)$ around the point $x_i$

To find $\alpha_i$, we will solve a linear system formed by setting $f(x)$ to $f(x_i)$ in our above equation for all control points $x_i$. For example, we have 3 control point correspondences $(x_0, x'_0)$, $(x_1, x'_1)$,  $(x_2, x'_2)$, corresponding weights $\alpha_0$, $\alpha_1$, $\alpha_2$ will be the solution to the following linear system
$$
\left[ \begin{array}{c}
x'_0 \\
x'_1 \\
x'_2
s\end{array} \right] =
\begin{bmatrix}
R(x_0, x_0) & R(x_0, x_1) & R(x_0, x_2)\\
R(x_1, x_0) & R(x_1, x_1) & R(x_0, x_2)\\
R(x_2, x_0) & R(x_2, x_1) & R(x_0, x_2)
\end{bmatrix} \times \left[
\begin{array}{c}
\alpha_0 \\
\alpha_1 \\
\alpha_2
\end{array} \right]
$$
The solution to the system will be
$$
\vec{\alpha} = R^{-1}\times \vec{x}
$$

The possible radial basis functions could be a gaussian kernel $e^{\frac{-r^2}{2\sigma}}$ or $r^2 log(r)$, which is the thin-plate-spline function for the $1D$ case.

# Thin Plate Spline

## What to solve for?
In the below figures, we refine a bit further the definition of the thin plate spline warping problem. Given red cross points the input images that are expected to move to blue circle points, we want to solve for two smooth functions from which we can sample for discrete displacements along $x$ and $y$ directions, as depicted by the red and blue arrows in the below figure.

![](/assets/images/tps/2019-d278d8c0.png)

The two smooth functions are shown in below, as modified from the original equations in [the paper](https://pdfs.semanticscholar.org/d926ea562d1de8143b0fe119b9a772cdef8cc50e.pdf)

$$
f_{x'}(x,y) = a_1 + a_xx +a_yy + \sum_{i=1}^N{w_i U(||(x_i, y_i) - (x,y)||)}
\newline
f_{y'}(x,y) = a_1 + a_xx +a_yy + \sum_{i=1}^N{w_i U(||(x_i, y_i) - (x,y)||)}
$$

With these two functions, we can pass in $(x,y)$ values of black grid points in the input image, take its displaced points output $(x', y')$ and calculate the corresponding $\Delta_x$ and $\Delta_y$ displacements, as visualized by the height of the surface in the two below figures.

![](/assets/images/tps/2019-4b0c6ba8.png)

As explained in the article by [hlombaert](https://profs.etsmtl.ca/hlombaert/thinplates/),

The three first coefficients $(a_1, a_x, a_y)$ represents the linear plane that best approximates all $x'$ (or $y'$) of all control points $(x_i, y_i)$.

The later terms $w_i$ for $i \in [0,N)$ denotes the weight of each control point to the final displacement.

The terms $U(||(x_i, y_i) - (x,y))||)$ represent the distance from the point $(x,y)$ to kernel of the control point $(x_i, y_i)$. The visualization of this thin plate spline kernel $U(r) = r^2log(r)$ is shown in below. The closer a point to the kernel center (one control point), the higher its height or the return value is.

![](/assets/images/tps/2019-8076195d.png)

The below visualization of a complex surface could be used to imagine how all the kernels look like when they are multiplied by their corresponding weights $w_i$
![](/assets/images/tps/2019-18b0ca24.png)

## how to solve
Coefficients $(a, a_x, a_y, w_i)$ of each thin plate spline function are found by solving the following linear system

$$
L =
\begin{bmatrix}K &   P \\ P^T & O \\ \end{bmatrix}
\times
\begin{bmatrix}w\\a\end{bmatrix}
=\begin{bmatrix}v\\o\end{bmatrix}
$$

where $K_{ij} = U(||(x_i,x_x) - (x_j, y_j)||)$, the ith of P is $(1, x_i, y_i)$.

The top row of the matrix $[K \ \  P]$ $\times$ $[v, o]^T$ represents the function $f_{\Delta}$ with all control points $(x_i, y_i)$ substituted.

The second row of the matrix $[P^T\ \  O]$ represents constraints for the function to be solved.

$$
\sum_{i=1}^N{w_i} = 0
\newline
\sum_{i=1}^N{w_ix_i} = \sum_{i=1}^N{w_iy_i} = 0
$$

To make it clear, we will build two linear system to solve for coefficients of $f_\Delta{x}$ and $f_\Delta{y}$ given 3 control point correspondences.

The linear system for the TPS function $f_{x'} $ will look like as follows.

$$
\begin{bmatrix}
K_{00} & K_{10} &  K_{20}&1&x_0&y_0 \\
K_{01} & K_{11} &  K_{21}&1&x_1&y_1 \\
K_{02} & K_{12} &  K_{22}&1&x_2&y_2 \\
1 & 1 &  1&0&0&0\\
x_0&x_1&x_2&0&0&0\\
y_0&y_1&y_2&0&0&0\\
\end{bmatrix}
\times
\begin{bmatrix}
w_0\\w_1\\w_2\\a_0\\a_x\\a_y
\end{bmatrix}
=\begin{bmatrix}x'_0\\x'_1\\x'_2\\0\\0\\0\end{bmatrix}
$$

and the TPS function $f_{y'}$

$$
\begin{bmatrix}
K_{00} & K_{10} &  K_{20}&1&x_0&y_0 \\
K_{01} & K_{11} &  K_{21}&1&x_1&y_1 \\
K_{02} & K_{12} &  K_{22}&1&x_2&y_2 \\
1 & 1 &  1&0&0&0\\
x_0&x_1&x_2&0&0&0\\
y_0&y_1&y_2&0&0&0\\
\end{bmatrix}
\times
\begin{bmatrix}
w_0\\w_1\\w_2\\a_0\\a_x\\a_y
\end{bmatrix}
=\begin{bmatrix}y'_0\\y'_1\\y'_2\\0\\0\\0\end{bmatrix}
$$
# OpenCV Thin Plate Spline implementation

## Solve for coefficients of $f_{x'}$ and $f_{y'}$

### The Thin Plate Spline Kernel function

$$U(r) = r^2log(r)$$

```c++
static float distance(Point2f p, Point2f q)
{
    Point2f diff = p - q;
    float norma = diff.x*diff.x + diff.y*diff.y;// - 2*diff.x*diff.y;
    if (norma<0) norma=0;
    //else norma = std::sqrt(norma);
    norma = norma*std::log(norma+FLT_EPSILON);
    return norma;
}
```
### Build matrix L

$$
L =
\begin{bmatrix}K &   P \\ P^T & O \\ \end{bmatrix}
\times
\begin{bmatrix}w\\a\end{bmatrix}
=\begin{bmatrix}v\\o\end{bmatrix}
$$

```C++
// Building the matrices for solving the L*(w|a)=(v|0) problem with L={[K|P];[P'|0]}
// Building K and P (Needed to build L)
Mat matK((int)matches.size(),(int)matches.size(),CV_32F);
Mat matP((int)matches.size(),3,CV_32F);
for (int i=0, end=(int)matches.size(); i<end; i++)
{
    for (int j=0; j<end; j++)
    {
        if (i==j)
        {
            // in our setting, it K[i,i] = 0, but here opencv add regulerization factor
            matK.at<float>(i,j)=float(regularizationParameter);
        }
        else
        {
            // calculate distance from point j to the TSP kernel i
            matK.at<float>(i,j) = distance(Point2f(shape1.at<float>(i,0),shape1.at<float>(i,1)),
                                           Point2f(shape1.at<float>(j,0),shape1.at<float>(j,1)));
        }
    }

    // set point (x_i, y_i) to the matrix P
    matP.at<float>(i,0) = 1;
    matP.at<float>(i,1) = shape1.at<float>(i,0);
    matP.at<float>(i,2) = shape1.at<float>(i,1);
}

// Building L
// Copy K, P, P^T blocks to the matrix L
Mat matL=Mat::zeros((int)matches.size()+3,(int)matches.size()+3,CV_32F);
Mat matLroi(matL, Rect(0,0,(int)matches.size(),(int)matches.size())); //roi for K
matK.copyTo(matLroi);
matLroi = Mat(matL,Rect((int)matches.size(),0,3,(int)matches.size())); //roi for P
matP.copyTo(matLroi);
Mat matPt;
transpose(matP,matPt);
matLroi = Mat(matL,Rect(0,(int)matches.size(),(int)matches.size(),3)); //roi for P'
matPt.copyTo(matLroi);

```

### Build the right hand side $[v\ o]^T$
$$
L =
\begin{bmatrix}K &   P \\ P^T & O \\ \end{bmatrix}
\times
\begin{bmatrix}w\\a\end{bmatrix}
=\begin{bmatrix}v\\o\end{bmatrix}
$$


```c++
//Building B (v|0)
Mat matB = Mat::zeros((int)matches.size()+3,2,CV_32F);
for (int i=0, end = (int)matches.size(); i<end; i++)
{
    matB.at<float>(i,0) = shape2.at<float>(i,0); //x's
    matB.at<float>(i,1) = shape2.at<float>(i,1); //y's
}
```

### Solve for $f_{x'}$ and $f_{x'}$
```c++
//Obtaining transformation params (w|a)
solve(matL, matB, tpsParameters, DECOMP_LU);
//tpsParameters = matL.inv()*matB;

//Setting transform Cost and Shape reference
Mat w(tpsParameters, Rect(0,0,2,tpsParameters.rows-3));
Mat Q=w.t()*matK*w;
transformCost=fabs(Q.at<float>(0,0)*Q.at<float>(1,1));//fabs(mean(Q.diag(0))[0]);//std::max(Q.at<float>(0,0),Q.at<float>(1,1));
tpsComputed=true;
```
## Warp image using estimated $f_{x'}$ and $f_{y'}$

### Build remap data structure
First a remap data structure will constructed by sampling from our continuous functions $f_{\Delta{x}}$ and $f_{\Delta{y}}$. This data structure stores displaced locations in the warped image for each origin locations in the input image. A function called $remap$ will use this data to interpolate colors for pixels in the warped image.

```c++
/* public methods */
void ThinPlateSplineShapeTransformerImpl::warpImage(InputArray transformingImage, OutputArray output, int flags, int borderMode, const Scalar& borderValue) const
{
    CV_INSTRUMENT_REGION();

    CV_Assert(tpsComputed==true);

    Mat theinput = transformingImage.getMat();
    Mat mapX(theinput.rows, theinput.cols, CV_32FC1);
    Mat mapY(theinput.rows, theinput.cols, CV_32FC1);

    // for each pixel in the input image
    for (int row = 0; row < theinput.rows; row++)
    {
        // for each pixel in the input image
        for (int col = 0; col < theinput.cols; col++)
        {
            // sample the corresponding location in the warped image space
            Point2f pt = _applyTransformation(shapeReference, Point2f(float(col), float(row)), tpsParameters);
            mapX.at<float>(row, col) = pt.x;
            mapY.at<float>(row, col) = pt.y;
        }
    }
    remap(transformingImage, output, mapX, mapY, flags, borderMode, borderValue);
}
```

### Sample $f_{x'}$ and $f_{y'}$ given a point $(x,y)$

Below is the implementation of the function $\_applyTransformation$ that is called from the above code snippet.

```c++
static Point2f _applyTransformation(const Mat &shapeRef, const Point2f point, const Mat &tpsParameters)
{
    Point2f out;
    // sample x' and y'
    // i == 0 => sample x'
    // i == 1 => sample y'
    for (int i=0; i<2; i++)
    {
        // calculate affine term: a1 + a_x*x + a_y*y
        // from the corresponding tps function
        float a1=tpsParameters.at<float>(tpsParameters.rows-3,i);
        float ax=tpsParameters.at<float>(tpsParameters.rows-2,i);
        float ay=tpsParameters.at<float>(tpsParameters.rows-1,i);

        float affine=a1+ax*point.x+ay*point.y;

        // sum up contributions from all kernels, given the kernel weights
        // from the corresponding tps function
        float nonrigid=0;
        for (int j=0; j<shapeRef.rows; j++)
        {
            nonrigid+=tpsParameters.at<float>(j,i)*
                    distance(Point2f(shapeRef.at<float>(j,0),shapeRef.at<float>(j,1)),
                            point);
        }

        //set sampled x and y to the output
        if (i==0)
        {
            out.x=affine+nonrigid;
        }
        if (i==1)
        {
            out.y=affine+nonrigid;
        }
    }
    return out;
}
```

</b>
</b>
</b>
</b>
</b>
</b>
</b>
fdasfdsa
fasdfd
img_as_floatdasf
fdasfdsadasf
as
fds
