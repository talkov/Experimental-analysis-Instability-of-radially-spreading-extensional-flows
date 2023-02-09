# Experimental-analysis-Instability-of-radially-spreading-extensional-flows
Code and article of the Experimental analysis to the Instability of radially spreading extensional flows experiments done by Dr. Roiy Sayag.



<h2>Description</h2>
The project focuses on the analysis of experimental data obtained during the
laboratory experiment described in the great paper: “Instability of radially
spreading extensional flows. Part 1. Experimental analysis J. Fluid
(2019)Cambridge University Press” by Dr. Roiy Sayag and Dr. M. Grae Worster. 
<br />
https://core.ac.uk/download/pdf/237398633.pdf
<br />

 
<h2>Languages and Utilities Used</h2>
- <b>Python</b> 



<h2>Code walk-through:</h2>
In general, the code uses image processing and Fourier decomposition (via the
FFT algorithm) of the fluid–fluid interface, to extract the wave number
associated with the most dominant (largest in absolute value) Fourier
coefficient.
<br />
<br />
-<b> Image processing </b>
<br />
By using the OPENCV, python library, each image in the data set is
converted to HSV format, masked and resized as preparation for the ”draw
contour” process.
By using the cv2.findContours function, a dictionary of possible contours is
created. The wanted contour (in our case the one that encapsulates the largest
area) is then selected and drawn on a copy of the original image and saved as
the main contour for the rest of the analysis.
<br />
<br />

-<b> Creating axis and function R(θ) </b>
<br />
After obtaining the main contour, that is represented by a vector of (x,y)
points in the processed image matrix, each point is processed to yield two new
vectors. The first vector represents the distance R from the origin. The second
vector represents the angle θ made between the line segment from the origin to
each point and the positive x-axis.The angle is represented in degrees between
0 and 360. To represent the contour in polar coordinates (R,θ). The Cartesian
axis origin is set to the pixel(267,252) in the original image. due to the
comfortable nature of the data set, it is the same for all the images in the data.
Creation of the vectors that represents the contour in polar coordinates is
possible by two methods that are selected by the user.
<br />

*<b>Option 1 </b>
<br />
The first option uses the complex plane and the python library Numpy. By
creating complex variable Z. <br />
<br />
*<b>Option 2 </b>
<br />
The second option uses the definition of the Cartesian-polar transformation.
<br />

To avoid unwanted behavior of the Fourier decomposition, due to harmonical oscillations which was caused by the tearing of the fluid near the end of the interface at each finger, the maximum radius was set to 60, thus the fingers were truncate to suggest the main shape of the fingers and avoid unwanted harmonics caused by the tearing.
<br />

-<b> Interpolating R(θ) function and “sampling” it in steady sampling frequency </b>
<br />
After the creation of the two vectors that represents the contour in polar
coordinates (R,θ),the vectors are sorted. A function is then interpolated based
on the data received from the two vectors. Using the numpy.interp function.
The interpolated function is then sampled by uniform sampling between 0 and
360 with intervals of 45/4096 between each sample.
The size of the interval was selected to best fit the FFT algorithm, that works
best when the input vector has a length that is a whole power of 2. The vector
is then sorted.

-<b> Fourier decomposition of the data </b>
<br />
After creating and sorting the “sample” vector, it ran through the FFT
algorithm using the numpy.fft.fft function which then returns the vector of
Fourier coefficient associated with the given data (the sampled interpolated
function). Each index in the coefficient vector, represents the wave number
that corresponds to the coefficient. Due to the Nyquist–Shannon sampling
theorem, the maximum wave number is N/2, where N is the number of
samples in the vector that is given to the algorithm.
<p align="center">



<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
